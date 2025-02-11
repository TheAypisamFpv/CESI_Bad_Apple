import cv2
import numpy as np
import pandas as pd
import torch
import json
import time
from trainNeuralNet import NeuralNet

predefinedWidth = 480
predefinedHeight = 360



def displayFrame(frameId, frame, width, height):
    # Resize the frame to a predefined size (e.g., 960x720)
    frame = cv2.resize(frame, (width, height))
    cv2.imshow('Video Frame', frame)
    cv2.waitKey(1)  # Display the frame for 1 ms

def main():
    modelPath = r'D:\VS_Python_Project\CESI_Bad_Apple\AI\models\24x18\goodStart\bestModel.pt'
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

    device = torch.device('cpu')
    
    # Load model
    model = torch.load(modelPath, weights_only=False)
    model = model.to(device)
    model.eval()
    print("Done\n")


    # Load input frames from CSV file
    print(f"Loading input frames from {dataCsvPath}...", end=' ')
    data_df = pd.read_csv(dataCsvPath)
    inputFrames = data_df['resizedFrameData'].apply(lambda x: np.fromstring(x.strip('[]'), sep=',')).values
    print("Done")

    input("Press Enter to start the video...")

    numDigits = len(str(len(inputFrames)))

    channels = 1  # Grayscale

    frameId = 0
    fps = 30
    frameDuration = 1 / fps  # Duration in seconds for each frame
    try:
        while frameId < len(inputFrames):
            startTime = time.time()
            # Get the input frame
            inputFrame = inputFrames[frameId]
            # Reshape input frame using the original input dimensions
            inputFrameReshaped = inputFrame.reshape((int(inputHeight), int(inputWidth)))
            inputTensor = torch.from_numpy(inputFrame).pin_memory().unsqueeze(0).float().to(device, non_blocking=True)
            
            with torch.no_grad():
                prediction = model(inputTensor).cpu().numpy()* 50.0
    
            # Reshape the predicted output to an image
            predictedFrame = prediction.reshape((outputHeight, outputWidth, channels))
    
            # Display predicted frame
            displayFrame(frameId, predictedFrame, predefinedWidth, predefinedHeight)
    
            # Additionally, display input frame in a separate, smaller window.
            inputImage = cv2.resize(inputFrameReshaped, (predefinedWidth//2, predefinedHeight//2))
            cv2.imshow("Input Video", inputImage)

            # Calculate elapsed time and sleep for the remaining frame duration.
            elapsed = time.time() - startTime
            remaining = max(0, (frameDuration - elapsed))
            time.sleep(remaining)
            cv2.waitKey(1)  # minimal wait to allow window refresh


            print(f"Frame ID: {frameId:0{numDigits}d}  |  Prediction time: {elapsed:.3f}s  |  fps: {1 / (elapsed + remaining):.1f}", end='\r')
    
            # Increment frame ID for the next prediction
            frameId += 1
    except KeyboardInterrupt:
        pass
    finally:
        print("\nVideo playback stopped")
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()