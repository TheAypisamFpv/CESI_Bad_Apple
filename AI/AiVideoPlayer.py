import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import json

predefinedWidth = 480
predefinedHeight = 360

def displayFrame(frameId, frame, width, height):
    # Resize the frame to a predefined size (e.g., 960x720)
    frame = cv2.resize(frame, (width, height))
    cv2.imshow('Video Frame', frame)
    cv2.waitKey(1)  # Display the frame for 1 ms
class NeuralNet(nn.Module):
    def __init__(self, layers, dropoutRates, l2Reg, inputActivation, hiddenActivation, outputActivation):
        super(NeuralNet, self).__init__()
        self.layers = nn.ModuleList()
        self.dropoutRates = dropoutRates
        self.l2Reg = l2Reg

        # Add input layer
        self.layers.append(nn.Linear(layers[0], layers[1]))
        self.layers.append(self.getActivation(inputActivation))
        self.layers.append(nn.BatchNorm1d(layers[1]))
        self.layers.append(nn.Dropout(dropoutRates[0]))

        # Add hidden layers
        for i in range(1, len(layers) - 2):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
            self.layers.append(self.getActivation(hiddenActivation))
            self.layers.append(nn.BatchNorm1d(layers[i + 1]))
            dropoutRateIndex = min(i, len(dropoutRates) - 1)
            self.layers.append(nn.Dropout(dropoutRates[dropoutRateIndex]))

        # Add output layer
        self.layers.append(nn.Linear(layers[-2], layers[-1]))
        self.layers.append(self.getActivation(outputActivation))

    def getActivation(self, activation):
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'sigmoid':
            return nn.Sigmoid()
        elif activation == 'tanh':
            return nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

    def forward(self, x):
        for layer in self.layers:
            if isinstance(layer, nn.BatchNorm1d):
                if x.size(0) == 1:
                    continue  # Skip batch normalization if batch size is 1
            x = layer(x)
        return x

def main():
    modelPath = r'D:\VS_Python_Project\CESI_Bad_Apple\AI\models\24x18\bestModel.pt'
    paramsPath = r'D:\VS_Python_Project\CESI_Bad_Apple\AI\models\24x18\bestModelParams.json'
    sizeCsvPath = r'D:\VS_Python_Project\CESI_Bad_Apple\AI\Bad Apple!!_size.csv'
    dataCsvPath = r'D:\VS_Python_Project\CESI_Bad_Apple\AI\Bad Apple!!_24x18.csv'

    # Load model parameters from JSON file
    print(f"Loading model parameters from {paramsPath}...", end=' ')
    with open(paramsPath, 'r') as f:
        bestParams = json.load(f)
    print("Done")

    # Load frame sizes from CSV file
    size_df = pd.read_csv(sizeCsvPath)
    inputWidth = size_df['initialNewFrameWidth'].iloc[0]
    inputHeight = size_df['initialNewFrameHeight'].iloc[0]
    outputWidth = size_df['outputNewFrameWidth'].iloc[0]
    outputHeight = size_df['outputNewFrameHeight'].iloc[0]

    print(f"Input frame size: {inputWidth}x{inputHeight}")
    print(f"Output frame size: {outputWidth}x{outputHeight}")

    print("Loading model...", end=' ')
    # Load model
    model = NeuralNet(
        layers=bestParams['layers'],
        dropoutRates=bestParams['dropoutRates'],
        l2Reg=bestParams['l2Reg'],
        inputActivation=bestParams['inputActivation'],
        hiddenActivation=bestParams['hiddenActivation'],
        outputActivation=bestParams['outputActivation']
    )
    state_dict = torch.load(modelPath, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    print("Done\n")


    # Load input frames from CSV file
    print(f"Loading input frames from {dataCsvPath}...", end=' ')
    data_df = pd.read_csv(dataCsvPath)
    inputFrames = data_df['resizedFrameData'].apply(lambda x: np.fromstring(x.strip('[]'), sep=',')).values
    print("Done")

    input("Press Enter to start the video...")


    channels = 1  # Grayscale

    frameId = 0
    fps = 30
    frameSkip = 1
    try:
        while frameId < len(inputFrames):
            frameStart = cv2.getTickCount()
            # Get the input frame
            inputFrame = inputFrames[frameId]
            # Reshape input frame using the original input dimensions
            inputFrameReshaped = inputFrame.reshape((int(inputHeight), int(inputWidth)))
            inputTensor = torch.tensor([inputFrame], dtype=torch.float32)
            
            with torch.no_grad():
                prediction = model(inputTensor).numpy()
    
            # Reshape the predicted output to an image
            predictedFrame = prediction.reshape((outputHeight, outputWidth, channels))
    
            # Display predicted frame
            print(f"Frame ID: {frameId}")
            displayFrame(frameId, predictedFrame, predefinedWidth, predefinedHeight)
    
            # Additionally, display input frame in a separate, smaller window.
            inputImage = cv2.resize(inputFrameReshaped, (predefinedWidth//2, predefinedHeight//2))
            cv2.imshow("Input Video", inputImage)

            # Calculate the time taken to process the frame, and how much time to wait before displaying the next frame
            frameEnd = cv2.getTickCount()
            frameTime = (frameEnd - frameStart) / cv2.getTickFrequency()
            waitTime = max(1, int(1000 / fps - frameTime * 1000))
            key = cv2.waitKey(waitTime * frameSkip)
                
    
            # Increment frame ID for the next prediction
            frameId += 1
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()