import cv2
import numpy as np
import pandas as pd
import torch
import torch.serialization
import time
import os
from customModel import BadAppleModel

# Default display dimensions
DEFAULT_WIDTH = 480
DEFAULT_HEIGHT = 360

def loadData(datasetPath):
    """
    Load data from a CSV file.
    
    Args:
        datasetPath: Path to the CSV file containing frames
        
    Returns:
        Pandas DataFrame with the data
    """
    print(f"Loading data from {datasetPath}")
    data = pd.read_csv(datasetPath)
    
    # Convert string representation of lists to numpy arrays
    data['resizedFrameData'] = data['resizedFrameData'].apply(lambda x: np.fromstring(x.strip('[]'), sep=','))
    if 'originalFrameData' in data.columns:
        data['originalFrameData'] = data['originalFrameData'].apply(lambda x: np.fromstring(x.strip('[]'), sep=','))
    
    return data

def displayFrame(frame, width, height, windowName='Video Frame'):
    """
    Display a frame in a window with the given name.
    """
    # Reshape the frame to proper dimensions if it's flattened
    if len(frame.shape) == 1:
        # Try to determine the aspect ratio if not provided
        # This is a simple approach - for more accurate dimensions, use sizeCsvPath
        side = int(np.sqrt(frame.shape[0]))
        frame = frame.reshape((side, frame.shape[0] // side))
    
    # Normalize if not already in 0-255 range
    if frame.max() <= 1.0:
        frame = frame * 255
    
    # Convert to uint8 for OpenCV
    frame = frame.astype(np.uint8)
    
    # Resize the frame to the specified dimensions
    frame = cv2.resize(frame, (width, height))
    
    # Show the frame
    cv2.imshow(windowName, frame)
    return cv2.waitKey(1)  # Return key pressed (if any)

def testModel(modelPath, datasetPath, sizeCsvPath=None, numSamples=5):
    """
    Test the trained model on a few samples from the dataset.
    
    Args:
        modelPath: Path to the trained model file
        datasetPath: Path to the dataset CSV file
        sizeCsvPath: Path to the CSV file containing frame dimensions
        numSamples: Number of random samples to test
    """
    # Load model
    print(f"Loading model from {modelPath}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Add BadAppleModel to safe globals for loading
    torch.serialization.add_safe_globals(['customModel.BadAppleModel'])
    
    try:
        # Try loading with weights_only=True (secure but might fail)
        model = torch.load(modelPath, map_location=device)
    except Exception:
        print("Attempting to load model with weights_only=False (legacy mode)...")
        # Fall back to weights_only=False (less secure but more compatible)
        model = torch.load(modelPath, map_location=device, weights_only=False)
    
    model.eval()
    
    # Load frame dimensions
    inputWidth, inputHeight, outputWidth, outputHeight = loadFrameDimensions(sizeCsvPath)
    print(f"Input dimensions: {inputWidth}x{inputHeight}")
    print(f"Output dimensions: {outputWidth}x{outputHeight}")
    
    # Load test data
    data = loadData(datasetPath)
    
    # Choose random samples
    indices = np.random.choice(len(data), size=min(numSamples, len(data)), replace=False)
    testData = data.iloc[indices]
    
    # Convert to numpy arrays
    inputFrames = np.stack(testData['resizedFrameData'].values)
    outputFrames = np.stack(testData['originalFrameData'].values)
    
    print("\nRunning model predictions...")
    
    with torch.no_grad():
        for i in range(len(inputFrames)):
            inputFrame = inputFrames[i]
            outputFrame = outputFrames[i]
            
            # Prepare input tensor
            inputTensor = torch.tensor(inputFrame, dtype=torch.float32).unsqueeze(0).to(device)
            
            # Make prediction
            startTime = time.time()
            prediction = model(inputTensor).cpu().numpy().squeeze()
            endTime = time.time()
            
            print(f"\nSample {i+1}/{len(inputFrames)}")
            print(f"Prediction time: {(endTime - startTime)*1000:.2f}ms")
            
            # Calculate error metrics
            mse = np.mean((prediction - outputFrame) ** 2)
            mae = np.mean(np.abs(prediction - outputFrame))
            print(f"MSE: {mse:.6f}, MAE: {mae:.6f}")
            
            # Display frames side by side
            print("Displaying input, ground truth, and prediction side by side.")
            print("Press any key to continue to the next sample...")
            
            # Reshape frames
            inputImg = inputFrame.reshape((inputHeight, inputWidth))
            outputImg = outputFrame.reshape((outputHeight, outputWidth))
            predictedImg = prediction.reshape((outputHeight, outputWidth))
            
            # Normalize to 0-255 and convert to uint8
            inputImg = (inputImg * 255).astype(np.uint8)
            outputImg = (outputImg * 255).astype(np.uint8)
            predictedImg = (predictedImg * 255).astype(np.uint8)
            
            # Resize for display
            displayWidth, displayHeight = 400, 300
            inputImg = cv2.resize(inputImg, (displayWidth, displayHeight))
            outputImg = cv2.resize(outputImg, (displayWidth, displayHeight))
            predictedImg = cv2.resize(predictedImg, (displayWidth, displayHeight))
            
            # Create a horizontal stack of the three images
            displayImg = np.hstack([inputImg, outputImg, predictedImg])
            
            # Display the combined image
            cv2.imshow('Input | Ground Truth | Prediction', displayImg)
            cv2.waitKey(0)  # Wait for key press
    
    cv2.destroyAllWindows()
    print("\nTest complete.")

def loadFrameDimensions(sizeCsvPath=None):
    """
    Load or infer frame dimensions.
    
    Args:
        sizeCsvPath: Path to the CSV file containing frame dimensions
        
    Returns:
        Tuple of (inputWidth, inputHeight, outputWidth, outputHeight)
    """
    if sizeCsvPath and os.path.exists(sizeCsvPath):
        print(f"Loading frame dimensions from {sizeCsvPath}")
        size_df = pd.read_csv(sizeCsvPath)
        inputWidth = size_df['initialNewFrameWidth'].iloc[0]
        inputHeight = size_df['initialNewFrameHeight'].iloc[0]
        outputWidth = size_df['outputNewFrameWidth'].iloc[0]
        outputHeight = size_df['outputNewFrameHeight'].iloc[0]
    else:
        # Default dimensions
        print("Using default dimensions")
        inputWidth, inputHeight = 24, 18
        outputWidth, outputHeight = 120, 90
    
    return inputWidth, inputHeight, outputWidth, outputHeight

def playVideo(modelPath, datasetPath, sizeCsvPath=None, fps=30, displayWidth=DEFAULT_WIDTH, displayHeight=DEFAULT_HEIGHT):
    """
    Play the Bad Apple animation using the trained model.
    
    Args:
        modelPath: Path to the trained model file
        datasetPath: Path to the dataset CSV file
        sizeCsvPath: Path to the CSV file containing frame dimensions
        fps: Frames per second for playback
        displayWidth: Width of the display window
        displayHeight: Height of the display window
    """
    # Load model
    print(f"Loading model from {modelPath}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Add BadAppleModel to safe globals for loading
    torch.serialization.add_safe_globals(['customModel.BadAppleModel'])
    
    try:
        # Try loading with weights_only=True (secure but might fail)
        model = torch.load(modelPath, map_location=device)
    except Exception:
        print("Attempting to load model with weights_only=False (legacy mode)...")
        # Fall back to weights_only=False (less secure but more compatible)
        model = torch.load(modelPath, map_location=device, weights_only=False)
    
    model.eval()
    
    # Load frame dimensions
    inputWidth, inputHeight, outputWidth, outputHeight = loadFrameDimensions(sizeCsvPath)
    print(f"Input dimensions: {inputWidth}x{inputHeight}")
    print(f"Output dimensions: {outputWidth}x{outputHeight}")
    
    # Load input frames
    print(f"Loading input frames from {datasetPath}")
    data = loadData(datasetPath)
    inputFrames = np.stack(data['resizedFrameData'].values)
    
    print(f"Total frames: {len(inputFrames)}")
    input("Press Enter to start the video...")
    print()
    frameDuration = 1 / fps  # Duration in seconds for each frame
    numDigits = len(str(len(inputFrames)))
    frameId = 0
    
    try:
        while frameId < len(inputFrames):
            startTime = time.time()
            
            # Get the input frame
            inputFrame = inputFrames[frameId]
            
            # Convert to tensor for model input
            inputTensor = torch.tensor(inputFrame, dtype=torch.float32).unsqueeze(0).to(device)
            
            # Make prediction
            with torch.no_grad():
                prediction = model(inputTensor).cpu().numpy().squeeze()
            
            # Reshape the predicted output to an image
            predictedFrame = prediction.reshape((outputHeight, outputWidth))
            
            # Display predicted frame
            key = displayFrame(predictedFrame, displayWidth, displayHeight, 'Bad Apple')
            
            # Additionally, display input frame in a smaller window
            inputFrameReshaped = inputFrame.reshape((inputHeight, inputWidth))
            smallWidth = displayWidth // 4
            smallHeight = displayHeight // 4
            displayFrame(inputFrameReshaped, smallWidth, smallHeight, 'Input')
            
            # Calculate elapsed time and sleep for the remaining frame duration
            elapsed = time.time() - startTime
            remaining = max(0, frameDuration - elapsed)
            time.sleep(remaining)
            
            # Print status
            actualFps = 1 / (elapsed + remaining) if (elapsed + remaining) > 0 else 0
            print(f"Frame: {frameId+1:0{numDigits}d}/{len(inputFrames)}  |  "
                  f"Processing time: {elapsed*1000:.1f}ms  |  "
                  f"FPS: {actualFps:.1f}", end='\r')
            
            # Check for exit key (ESC or 'q')
            if key == 27 or key == ord('q'):
                print("\nPlayback stopped by user.")
                break
                
            # Increment frame ID for next prediction
            frameId += 1
            
    except KeyboardInterrupt:
        print("\nPlayback interrupted by user.")
    finally:
        print("\nVideo playback complete.")
        cv2.destroyAllWindows()

def main():
    # Settings configured directly in the code
    
    # Mode setting - choose 'play' or 'test'
    mode = 'play'  # Change to 'test' if you want to test the model instead of playing video
    
    # Common settings
    modelPath = r"models\BadAppleModel_24x18_To_480x360\badAppleModel_best.pt"
    datasetPath = r"dataset\24x18_To_480x360\Bad Apple!!_24x18.csv"
    sizeCsvPath = r"dataset\24x18_To_480x360\Bad Apple!!_size.csv"
    
    # Play mode settings
    fps = 30
    displayWidth = DEFAULT_WIDTH
    displayHeight = DEFAULT_HEIGHT
    
    # Test mode settings
    numSamples = 5
    
    print(f"Selected mode: {mode}")
    
    if mode == "play":
        playVideo(
            modelPath=modelPath,
            datasetPath=datasetPath,
            sizeCsvPath=sizeCsvPath,
            fps=fps,
            displayWidth=displayWidth,
            displayHeight=displayHeight
        )
    else:  # mode == "test"
        testModel(
            modelPath=modelPath,
            datasetPath=datasetPath,
            sizeCsvPath=sizeCsvPath,
            numSamples=numSamples
        )

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        cv2.destroyAllWindows()
