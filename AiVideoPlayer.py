import cv2
import numpy as np
import pandas as pd
import torch
import torch.serialization
import time
import os
import csv
import sys
import queue
import threading
from customModel import BadAppleModel

# Default display dimensions
DEFAULT_WIDTH = 480
DEFAULT_HEIGHT = 360


def parseFrameSeries(frame_str: str) -> np.ndarray:
    """Parse a CSV cell containing a list-like string of numbers into a 1D numpy array.

    This uses numpy.fromstring for speed and strips enclosing brackets if present.
    """
    # Guard against None or empty strings
    if frame_str is None or frame_str == "":
        return np.array([], dtype=np.float32)
    s = frame_str.strip()
    if s and s[0] == '[' and s[-1] == ']':
        s = s[1:-1]
    arr = np.fromstring(s, sep=',', dtype=np.float32)
    return arr

def streamFrames(datasetPath):
    """
    Stream frames from the dataset CSV without loading it entirely into memory.

    Yields dictionaries with keys:
      - 'resizedFrameData': 1D np.ndarray
      - 'originalFrameData': 1D np.ndarray or None if column not present
    """
    EnsureCsvFieldSizeLimit()
    with open(datasetPath, 'r', newline='') as f:
        reader = csv.DictReader(f)
        has_original = 'originalFrameData' in (reader.fieldnames or [])
        for row in reader:
            resized = parseFrameSeries(row.get('resizedFrameData'))
            original = parseFrameSeries(row.get('originalFrameData')) if has_original else None
            yield {
                'resizedFrameData': resized,
                'originalFrameData': original
            }

def EnsureCsvFieldSizeLimit():
    """Increase the csv.field_size_limit to handle very large fields.

    On some platforms, csv.field_size_limit(sys.maxsize) overflows; in that case,
    progressively reduce the value until it is accepted.
    """
    max_int = sys.maxsize
    while True:
        try:
            csv.field_size_limit(max_int)
            break
        except (OverflowError, ValueError):
            max_int = int(max_int / 10)
            if max_int <= 1024 * 1024:  # 1MB minimum safeguard
                csv.field_size_limit(1024 * 1024)
                break

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

def getFrameDimensions(sizeCsvPath=None):
    """
    Load or infer frame dimensions.
    
    Args:
        sizeCsvPath: Path to the CSV file containing frame dimensions
        
    Returns:
        Tuple of (inputWidth, inputHeight, outputWidth, outputHeight)
    """
    if sizeCsvPath and os.path.exists(sizeCsvPath):
        print(f"Loading frame dimensions from {sizeCsvPath}...", end="")
        size_df = pd.read_csv(sizeCsvPath)
        inputWidth = size_df['initialNewFrameWidth'].iloc[0]
        inputHeight = size_df['initialNewFrameHeight'].iloc[0]
        outputWidth = size_df['outputNewFrameWidth'].iloc[0]
        outputHeight = size_df['outputNewFrameHeight'].iloc[0]
        print("done.")
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
    print(f"Loading model from {modelPath}...", end="")
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
    print("done.")
    
    # Load frame dimensions
    inputWidth, inputHeight, outputWidth, outputHeight = getFrameDimensions(sizeCsvPath)
    print(f"Input dimensions: {inputWidth}x{inputHeight}")
    print(f"Output dimensions: {outputWidth}x{outputHeight}")
    
    # Inform about streaming mode
    print(f"Streaming frames from {datasetPath} with frame buffer")
    print("Display mode: Input | Prediction | Original")
    
    # Create a frame buffer using a queue and preload some frames
    frameBuffer = queue.Queue(maxsize=30)  # Buffer for ~10 seconds at 30fps
    stopEvent = threading.Event()
    
    # Define a function to preload frames in a separate thread
    def frameLoader():
        frameCount = 0
        for record in streamFrames(datasetPath):
            if stopEvent.is_set():
                break
                
            # Skip empty frames
            if record['resizedFrameData'].size == 0:
                continue
                
            # Try to add the frame to the buffer, wait if buffer is full
            try:
                frameBuffer.put(record, timeout=0.5)
                frameCount += 1
            except queue.Full:
                # If buffer is full, wait briefly before trying again
                time.sleep(0.01)
                continue
        
        # Add a sentinel to signal end of data
        frameBuffer.put(None)
        print(f"\nTotal frames processed: {frameCount}")
    
    # Start the frame loader thread
    loaderThread = threading.Thread(target=frameLoader, daemon=True)
    loaderThread.start()
    
    # Wait for buffer to fill initially
    print("Preloading frames into buffer...", end="")
    preloadCount = min(60, frameBuffer.maxsize // 2)  # Aim for 2 seconds or half the buffer
    while frameBuffer.qsize() < preloadCount and loaderThread.is_alive():
        time.sleep(0.1)
        if frameBuffer.qsize() > 0 and frameBuffer.qsize() % 10 == 0:
            print(".", end="", flush=True)
    print(f" done. ({frameBuffer.qsize()} frames ready)")
    
    input("Press Enter to start the video...")
    print()
    
    # Timing variables
    frameDuration = 1 / fps  # Target duration per frame in seconds
    frameId = 0
    videoStartTime = time.time()  # Overall video start time
    frameStartTime = videoStartTime  # Start time for current frame
    fpsUpdateInterval = 0.5  # Update FPS display every 0.5 seconds
    nextFpsUpdate = videoStartTime + fpsUpdateInterval
    
    # Track recent frame times for averaging FPS
    frameTimes = []
    maxFrameHistory = 30  # Keep the last 30 frame times for FPS calculation
    
    try:
        while True:
            # Start timing this frame
            frameStartTime = time.time()
            
            # Get next frame from buffer
            record = frameBuffer.get()
            
            # Check if we've reached the end of data
            if record is None:
                print("\nEnd of frames reached")
                break
            
            # Get the input frame
            inputFrame = record['resizedFrameData']
            if inputFrame.size == 0:
                continue
            
            # Convert to tensor for model input
            inputTensor = torch.tensor(inputFrame, dtype=torch.float32).unsqueeze(0).to(device)
            
            # Make prediction
            with torch.no_grad():
                prediction = model(inputTensor).cpu().numpy().squeeze()
            
            # Reshape frames
            inputFrameReshaped = inputFrame.reshape((inputHeight, inputWidth))
            predictedFrame = prediction.reshape((outputHeight, outputWidth))
            
            # Get original frame if available
            originalSeries = record.get('originalFrameData')
            if originalSeries is not None and originalSeries.size > 0:
                originalFrame = originalSeries.reshape((outputHeight, outputWidth))
            else:
                # If ground truth is not available, use a blank frame of the same size
                originalFrame = np.zeros((outputHeight, outputWidth), dtype=np.float32)
            
            inputResized = cv2.resize(inputFrameReshaped, (outputWidth, outputHeight))
            
            # Create a composite image with all three frames side by side
            # Border between frames
            borderWidth = 3
            borderColor = 128
            border = np.ones((outputHeight, borderWidth), dtype=np.float32) * (borderColor / 255.0)
            
            # Combine frames horizontally: Input | Border | Prediction | Border | Original
            combinedFrame = np.hstack([inputResized, border, predictedFrame, border, originalFrame])
            
            key = displayFrame(combinedFrame, 
                              width=(outputWidth * 3 + borderWidth * 2), 
                              height=outputHeight,
                              windowName='Bad Apple: Input | Prediction | Original')
            
            # Track frame processing time for FPS calculation
            processingTime = time.time() - frameStartTime
            
            # Calculate when this frame should be shown to maintain target FPS
            targetFrameTime = videoStartTime + (frameId * frameDuration)
            currentTime = time.time()
            
            # Time to wait to maintain target FPS
            sleepTime = max(0, targetFrameTime - currentTime)
            
            if sleepTime > 0:
                time.sleep(sleepTime)
                
            # Calculate actual frame time including sleep
            totalFrameTime = time.time() - frameStartTime
            frameTimes.append(totalFrameTime)
            
            # Keep only recent frame times
            if len(frameTimes) > maxFrameHistory:
                frameTimes.pop(0)
                
            # Calculate actual FPS based on average of recent frame times
            avgFrameTime = sum(frameTimes) / len(frameTimes)
            actualFps = 1.0 / avgFrameTime if avgFrameTime > 0 else 0
            
            # Print status
            currentTime = time.time()
            # if currentTime >= nextFpsUpdate:
            bufferStatus = f"Buffer: {frameBuffer.qsize()}/{frameBuffer.maxsize}"
            processingTime = f"Processing: {processingTime*1000:.1f}ms"
            fpsStatus = f"FPS: {actualFps:.1f}"
            print(f"Frame: {frameId+1}  |  {processingTime}  |  {fpsStatus}  |  {bufferStatus}", end='\r')
            nextFpsUpdate = currentTime + fpsUpdateInterval
            
            # Check for exit key (ESC or 'q')
            if key == 27 or key == ord('q'):
                print("\nPlayback stopped by user.")
                break
                
            # Increment frame ID for next prediction
            frameId += 1
            
    except KeyboardInterrupt:
        print("\nPlayback interrupted by user.")
    finally:
        # Signal the loader thread to stop and wait for it to finish
        stopEvent.set()
        if loaderThread.is_alive():
            loaderThread.join(timeout=1.0)
        
        # Show playback statistics
        totalTime = time.time() - videoStartTime
        avgFps = frameId / totalTime if totalTime > 0 else 0
        print(f"\nVideo playback complete. Frames: {frameId}, Time: {totalTime:.2f}s, Average FPS: {avgFps:.1f}")
        cv2.destroyAllWindows()

def main():
    # Settings configured directly in the code
        
    # Common settings
    modelPath = r"models\BadAppleModel_48x36_To_360x270_conv\badAppleModel_best.pt"
    datasetPath = r"dataset\48x36_To_360x270\Bad Apple!!_48x36.csv"
    sizeCsvPath = r"dataset\48x36_To_360x270\Bad Apple!!_size.csv"

    # Play mode settings
    fps = 30
    displayWidth = DEFAULT_WIDTH
    displayHeight = DEFAULT_HEIGHT
     
    playVideo(
        modelPath=modelPath,
        datasetPath=datasetPath,
        sizeCsvPath=sizeCsvPath,
        fps=fps,
        displayWidth=displayWidth,
        displayHeight=displayHeight
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        cv2.destroyAllWindows()
