import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Residual block with skip connection to preserve features during upscaling.
    """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity  # Skip connection
        out = self.relu(out)
        return out


class UpsampleBlock(nn.Module):
    """
    Upsampling block to increase spatial dimensions.
    """
    def __init__(self, inChannels, outChannels):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Conv2d(inChannels, outChannels * 4, kernel_size=3, padding=1)
        self.pixelShuffle = nn.PixelShuffle(2)  # Increases spatial dimensions by 2x
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixelShuffle(x)
        x = self.relu(x)
        return x


class BadAppleModel(nn.Module):
    """
    Neural network model for upscaling Bad Apple frames from low to high resolution.
    Supports both 1D (flattened frames) and 2D (proper images) inputs.
    Automatically adapts to different input and output dimensions.
    """
    def __init__(self, inputSize: tuple, outputSize: tuple, useConvLayers=True):
        """
        Initialize the BadAppleModel with flexible dimensions.
        
        Args:
            inputSize: tuple (width, height) of the input frames
            outputSize: tuple (width, height) of the output frames
            useConvLayers: Whether to use convolutional layers (True) or fully connected layers (False)
        """
        super(BadAppleModel, self).__init__()
        
        self.inputWidth, self.inputHeight = inputSize
        self.inputSize = self.inputWidth * self.inputHeight
        
        self.outputWidth, self.outputHeight = outputSize
        self.outputSize = self.outputWidth * self.outputHeight
       
        
        self.useConvLayers = useConvLayers
        self.inputChannels = 1  # Grayscale images

        print(f"Model configured with input dimensions: {self.inputWidth}x{self.inputHeight}")
        print(f"Model configured with output dimensions: {self.outputWidth}x{self.outputHeight}")

        if useConvLayers:
            # Calculate the scale factor between input and output
            self.widthScale = self.outputWidth / self.inputWidth
            self.heightScale = self.outputHeight / self.inputHeight
            
            # Calculate approximate overall scale factor (geometric mean)
            self.overallScaleFactor = (self.widthScale * self.heightScale) ** 0.5
            
            print(f"Width scale factor: {self.widthScale:.2f}, Height scale factor: {self.heightScale:.2f}")
            print(f"Overall scale factor: {self.overallScaleFactor:.2f}")
            
            # Determine feature dimensions based on input size
            # For smaller inputs, use fewer features to avoid overfitting
            baseFeatures = min(64, max(32, min(self.inputWidth, self.inputHeight)))
            
            # CNN Architecture - base convolution
            self.initialConv = nn.Conv2d(1, baseFeatures, kernel_size=3, padding=1)

            # Count of residual blocks scales with image complexity
            residualBlocksCount = max(2, min(6, int(self.overallScaleFactor * 0.8)))
            self.resBlocks = nn.Sequential(
                *[ResidualBlock(baseFeatures) for _ in range(residualBlocksCount)]
            )
            
            print(f"Using {residualBlocksCount} residual blocks with {baseFeatures} features")
            
            # Calculate the exact number of upsampling blocks needed
            self.upsampleLayers = nn.ModuleList()
            
            # Each upsampling block doubles the dimensions
            # We calculate how many 2x upscaling operations we need
            # For example: scale factor of 5 needs 2 upsampling blocks (2x2=4) plus final interpolation
            
            # Determine needed upsampling blocks
            currentWidth, currentHeight = self.inputWidth, self.inputHeight
            upscaleCount = 0
            
            while currentWidth * 2 <= self.outputWidth and currentHeight * 2 <= self.outputHeight:
                self.upsampleLayers.append(UpsampleBlock(baseFeatures, baseFeatures))
                currentWidth *= 2
                currentHeight *= 2
                upscaleCount += 1
                print(f"Added upscaling block {upscaleCount}: {currentWidth}x{currentHeight}")
            
            # Final output layer
            self.finalConv = nn.Conv2d(baseFeatures, 1, kernel_size=3, padding=1)
            self.sigmoid = nn.Sigmoid()  # Ensure output is between 0 and 1
            
            # Store the current dimensions after upsampling for final interpolation
            self.intermediateWidth = currentWidth
            self.intermediateHeight = currentHeight
            
            print(f"Final interpolation will be from {self.intermediateWidth}x{self.intermediateHeight} to {self.outputWidth}x{self.outputHeight}")
            
        else:
            # Fully connected architecture for flattened inputs/outputs
            # Scale hidden layer sizes based on input and output dimensions
            
            # Calculate appropriate hidden layer sizes based on the data scale
            scaleRatio = self.outputSize / self.inputSize
            
            # Base hidden size on geometric mean of input and output, with a scaling factor
            hiddenSize1 = int(max(256, (self.inputSize * self.outputSize)**0.45))
            
            # Scale second hidden layer based on upscaling ratio
            hiddenSize2 = int(hiddenSize1 * min(4, max(1.5, scaleRatio**0.5)))
            
            print(f"FC Architecture: Input({self.inputSize}) -> Hidden1({hiddenSize1}) -> Hidden2({hiddenSize2}) -> Output({self.outputSize})")
            
            self.fc1 = nn.Linear(self.inputSize, hiddenSize1)
            self.bn1 = nn.BatchNorm1d(hiddenSize1)
            self.fc2 = nn.Linear(hiddenSize1, hiddenSize2)
            self.bn2 = nn.BatchNorm1d(hiddenSize2)
            self.fc3 = nn.Linear(hiddenSize2, hiddenSize2)
            self.bn3 = nn.BatchNorm1d(hiddenSize2)
            self.fc4 = nn.Linear(hiddenSize2, self.outputSize)
            
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batchSize = x.size(0)
        
        if self.useConvLayers:
            # Reshape from flattened to 2D if input is flattened
            if x.dim() == 2:
                x = x.view(batchSize, 1, self.inputHeight, self.inputWidth)
            
            # CNN forward pass
            x = self.initialConv(x)
            x = self.resBlocks(x)
            
            # Apply upsampling blocks
            for upsample in self.upsampleLayers:
                x = upsample(x)
            
            # Apply final convolution
            x = self.finalConv(x)
            
            # Resize to exact output dimensions using interpolation
            if (self.intermediateWidth != self.outputWidth or 
                self.intermediateHeight != self.outputHeight):
                x = F.interpolate(x, size=(self.outputHeight, self.outputWidth), 
                                 mode='bilinear', align_corners=False)
            
            # Apply sigmoid activation
            x = self.sigmoid(x)
            
            # Flatten the output to match the target shape [batch_size, outputSize]
            return x.view(batchSize, -1)
        else:
            # Fully connected forward pass
            x = self.relu(self.bn1(self.fc1(x)))
            x = self.relu(self.bn2(self.fc2(x)))
            x = self.relu(self.bn3(self.fc3(x)))
            x = self.sigmoid(self.fc4(x))
            return x

    def predict(self, x):
        """
        Forward pass with handling for different input formats.
        """
        self.eval()  # Set model to evaluation mode
        
        # Check if input is a torch tensor
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        
        # Add batch dimension if missing
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        with torch.no_grad():
            output = self.forward(x)
            
        return output


def createBadAppleModel(inputSize, outputSize, useConvLayers=True, device='cpu'):
    """
    Factory function to create and initialize the BadAppleModel.
    
    Args:
        inputSize: Either a tuple (width, height) or an integer for flattened size
        outputSize: Either a tuple (width, height) or an integer for flattened size
        useConvLayers: Whether to use convolutional layers (True) or fully connected layers (False)
        device: Device to place the model on ('cpu' or 'cuda')
        
    Returns:
        Initialized BadAppleModel
    """
    print(f"Creating BadAppleModel with input size: {inputSize}, output size: {outputSize}, convolutional: {useConvLayers}")
    model = BadAppleModel(inputSize, outputSize, useConvLayers)
    model = model.to(device)
    return model


def loadModel(modelPath, inputSize, outputSize, useConvLayers=True, device='cpu'):
    """
    Load a saved model from disk.
    
    Args:
        modelPath: Path to the saved model file
        inputSize: Either a tuple (width, height) or an integer for flattened input size
        outputSize: Either a tuple (width, height) or an integer for flattened output size
        useConvLayers: Whether the model uses convolutional layers
        device: Device to place the model on
        
    Returns:
        Loaded BadAppleModel
    """
    print(f"Loading BadAppleModel from {modelPath}")
    model = createBadAppleModel(inputSize, outputSize, useConvLayers, device)
    
    try:
        # Try loading with weights_only=True (secure but might fail)
        model.load_state_dict(torch.load(modelPath, map_location=device))
    except Exception:
        print("Attempting to load model with weights_only=False (legacy mode)...")
        # Fall back to weights_only=False (less secure but more compatible)
        model.load_state_dict(torch.load(modelPath, map_location=device, weights_only=False))
    
    model.eval()
    return model

