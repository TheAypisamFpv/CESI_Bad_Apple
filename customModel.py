import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------
# Dense + Residual-in-Residual blocks (ESRGAN-style)
# ------------------------------

class DenseBlock(nn.Module):
    """Dense block with 5 convolutional layers and local feature fusion."""
    def __init__(self, channels=64, growth=32):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(5):
            in_c = channels + i * growth
            self.layers.append(nn.Conv2d(in_c, growth, 3, 1, 1))
        self.fusion = nn.Conv2d(channels + 5 * growth, channels, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        inputs = [x]
        for conv in self.layers:
            out = self.lrelu(conv(torch.cat(inputs, 1)))
            inputs.append(out)
        out = self.fusion(torch.cat(inputs, 1))
        # Local residual scaling (0.2 for stability)
        return x + 0.2 * out


class RRDB(nn.Module):
    """Residual in Residual Dense Block"""
    def __init__(self, channels=64, growth=32):
        super().__init__()
        self.rdb1 = DenseBlock(channels, growth)
        self.rdb2 = DenseBlock(channels, growth)
        self.rdb3 = DenseBlock(channels, growth)

    def forward(self, x):
        out = self.rdb3(self.rdb2(self.rdb1(x)))
        return x + 0.2 * out


# ------------------------------
# Full Upscaling Network
# ------------------------------

class BadAppleRRDBModel(nn.Module):
    """
    RRDB-based model for high-quality upscaling of Bad Apple frames.
    Compatible with your existing training code.
    """

    def __init__(self, inputSize, outputSize, numBlocks=6, baseChannels=64, growth=32):
        super().__init__()
        self.inputWidth, self.inputHeight = inputSize
        self.outputWidth, self.outputHeight = outputSize
        self.inputChannels = 1
        self.outputChannels = 1

        # Initial feature extraction
        self.conv_first = nn.Conv2d(1, baseChannels, 3, 1, 1)

        # RRDB blocks
        self.RRDB_trunk = nn.Sequential(*[RRDB(baseChannels, growth) for _ in range(numBlocks)])
        self.trunk_conv = nn.Conv2d(baseChannels, baseChannels, 3, 1, 1)

        # Upsampling layers — 2x each
        # Determine upsample count dynamically
        scale_w = self.outputWidth / self.inputWidth
        scale_h = self.outputHeight / self.inputHeight
        numUpscale = int(torch.log2(torch.tensor(min(scale_w, scale_h))).item())
        self.upsample_layers = nn.ModuleList()
        for _ in range(numUpscale):
            self.upsample_layers.append(
                nn.Sequential(
                    nn.Conv2d(baseChannels, baseChannels * 4, 3, 1, 1),
                    nn.PixelShuffle(2),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )

        # Final reconstruction
        self.conv_last = nn.Conv2d(baseChannels, 1, 3, 1, 1)
        self.sigmoid = nn.Sigmoid()

        print(f"[RRDBModel] Input: {self.inputWidth}x{self.inputHeight}, "
              f"Output: {self.outputWidth}x{self.outputHeight}, "
              f"RRDB blocks: {numBlocks}, Upscale ×{2**numUpscale}")

    def forward(self, x):
        if x.dim() == 2:
            x = x.view(x.size(0), 1, self.inputHeight, self.inputWidth)

        fea = self.conv_first(x)
        trunk = self.RRDB_trunk(fea)
        fea = fea + self.trunk_conv(trunk)

        for up in self.upsample_layers:
            fea = up(fea)

        # Final interpolation to exact output size (if not power of 2 multiple)
        fea = self.conv_last(fea)
        if fea.shape[2:] != (self.outputHeight, self.outputWidth):
            fea = F.interpolate(fea, size=(self.outputHeight, self.outputWidth),
                                mode='bilinear', align_corners=False)

        out = self.sigmoid(fea)
        return out.view(x.size(0), -1)


def createBadAppleRRDBModel(inputSize, outputSize, device='cpu'):
    model = BadAppleRRDBModel(inputSize, outputSize)
    return model.to(device)
