import cv2
import numpy as np
import torch
import torch.serialization
import time
import os
import queue
import threading
from customModel import BadAppleModel

# Default display dimensions
DEFAULT_WIDTH = 480
DEFAULT_HEIGHT = 360

def displayFrame(frame, width, height, windowName='Video Frame'):
    """
    Display a frame in a window with the given name.
    """
    # Reshape the frame to proper dimensions if it's flattened
    if len(frame.shape) == 1:
        # Try to determine the aspect ratio if not provided
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

def playVideo(modelPath, videoPath, fps=30, displayWidth=DEFAULT_WIDTH, displayHeight=DEFAULT_HEIGHT, saveVideo=False):
    """
    Play the Bad Apple animation using the trained model directly from video file.
    
    Args:
        modelPath: Path to the trained model file
        videoPath: Path to the source video file
        fps: Frames per second for playback
        displayWidth: Width of the display window
        displayHeight: Height of the display window
        saveVideo: If True, saves the output video in the same directory as the model
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
    
    # Get model input/output dimensions from the model itself
    inputWidth, inputHeight = model.inputWidth, model.inputHeight
    outputWidth, outputHeight = model.outputWidth, model.outputHeight
    print(f"Model input dimensions: {inputWidth}x{inputHeight}")
    print(f"Model output dimensions: {outputWidth}x{outputHeight}")
    
    # Open the video file
    print(f"Opening video from {videoPath}...")
    cap = cv2.VideoCapture(videoPath)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return
    
    # Get video properties
    originalFps = cap.get(cv2.CAP_PROP_FPS)
    totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    originalWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    originalHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video properties: {originalWidth}x{originalHeight} at {originalFps:.2f} FPS, {totalFrames} frames")
    
    # Set up video writer if saveVideo is True
    videoWriter = None
    if saveVideo:
        # Create output directory in the same folder as the model
        modelDir = os.path.dirname(modelPath)
        outputVideoPath = os.path.join(modelDir, "badapple_output.mp4")

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        videoWriter = cv2.VideoWriter(
            outputVideoPath,
            fourcc,
            fps,
            ((outputWidth * 3 + 2 * 3), outputHeight),  # Width includes 3 frames and 2 borders
            isColor=True
        )
        print(f"Video will be saved to: {outputVideoPath}")
    
    # Set playback FPS if not specified
    if fps is None:
        fps = originalFps
    
    print("Display mode: Input | Prediction | Original")
    
    # Create a frame buffer using a queue and preload some frames
    frameBuffer = queue.Queue(maxsize=30)  # Buffer for ~10 seconds at 30fps
    stopEvent = threading.Event()
    
    # Define a function to preload frames in a separate thread
    def frameLoader():
        frameCount = 0
        while cap.isOpened() and not stopEvent.is_set():
            ret, frame = cap.read()
            if not ret:
                # End of video
                break
                
            # Convert frame to grayscale
            grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Resize to model input dimensions
            inputResized = cv2.resize(grayFrame, (inputWidth, inputHeight))
            
            # Normalize to [0,1]
            inputNormalized = inputResized / 255.0
            
            # Flatten for model input
            inputFlattened = inputNormalized.flatten()
            
            # Get original frame resized to output dimensions for comparison
            originalResized = cv2.resize(grayFrame, (outputWidth, outputHeight))
            originalNormalized = originalResized / 255.0
            
            # Try to add the frame to the buffer, wait if buffer is full
            try:
                frameBuffer.put({
                    'input': inputFlattened,
                    'original': originalNormalized
                }, timeout=0.5)
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
    print("Starting video playback in 3s...", end="\r")
    # Display a first frame to activate the display window
    # Use a loop to keep the window responsive during countdown
    startTime = time.time()
    while time.time() - startTime < 3:
        key = displayFrame(np.zeros((outputHeight, outputWidth * 3), dtype=np.float32),
                           outputWidth * 3,
                           outputHeight, 
                           windowName='Bad Apple: Input | Prediction | Original')

        print(f"Starting video playback in {3 - (time.time() - startTime):.1f}s...", end="\r")

        # Small sleep to prevent CPU overuse
        time.sleep(0.05)

    print("\n\n")

    # Timing variables
    frameDuration = 1 / fps  # Target duration per frame in seconds
    frameId = 0
    videoStartTime = time.time()  # Overall video start time
    frameStartTime = videoStartTime  # Start time for current frame
    fpsUpdateInterval = 10  # Update FPS display every 10 frames
    nextFpsUpdate = videoStartTime + fpsUpdateInterval
    
    # Track recent frame times for averaging FPS
    frameTimes = []
    maxFrameHistory = 30  # Keep the last 30 frame times for FPS calculation
    
    try:
        while True:
            # Start timing this frame
            frameStartTime = time.time()
            
            # Get next frame from buffer
            frameData = frameBuffer.get()
            
            # Check if we've reached the end of data
            if frameData is None:
                print("\nEnd of frames reached")
                break
            
            # Get the input frame
            inputFrame = frameData['input']
            originalFrame = frameData['original']
            
            # Convert to tensor for model input
            inputTensor = torch.tensor(inputFrame, dtype=torch.float32).unsqueeze(0).to(device)
            
            # Make prediction
            with torch.no_grad():
                prediction = model(inputTensor).cpu().numpy().squeeze()
            
            # Reshape frames
            inputFrameReshaped = inputFrame.reshape((inputHeight, inputWidth))
            predictedFrame = prediction.reshape((outputHeight, outputWidth))
            originalFrameReshaped = originalFrame.reshape((outputHeight, outputWidth))
            
            # Resize input to output dimensions for display
            inputResized = cv2.resize(inputFrameReshaped, (outputWidth, outputHeight))
            
            # Create a composite image with all three frames side by side
            # Border between frames
            borderWidth = 3
            borderGray = 128
            border = np.ones((outputHeight, borderWidth), dtype=np.float32) * (borderGray / 255.0)
            
            # Combine frames horizontally: Input | Border | Prediction | Border | Original
            combinedFrame = np.hstack([inputResized, border, predictedFrame, border, originalFrameReshaped])
            
            key = displayFrame(combinedFrame, 
                              width=(outputWidth * 3 + borderWidth * 2), 
                              height=outputHeight,
                              windowName='Bad Apple: Input | Prediction | Original')
            
            # Save frame to video if enabled
            if saveVideo and videoWriter is not None:
                # Convert to uint8 for video writing (0-255 range)
                frameToSave = (combinedFrame * 255).astype(np.uint8)
                # Convert grayscale to BGR (3-channel) for video writing
                frameToSaveBgr = cv2.cvtColor(frameToSave, cv2.COLOR_GRAY2BGR)
                videoWriter.write(frameToSaveBgr)
            
            # Track frame processing time for FPS calculation
            processingTime = time.time() - frameStartTime
            
            # Calculate when this frame should be shown to maintain target FPS
            targetFrameTime = videoStartTime + (frameId * frameDuration)
            
            # Time to wait to maintain target FPS
            sleepTime = max(0, targetFrameTime - time.time())
            
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
            if frameId % fpsUpdateInterval < 0.1:
                print(f"Frame: {frameId+1}/{totalFrames}  |  Processing: {processingTime*1000:.1f}ms  |  FPS: {actualFps:.1f}/{fps:.1f}  |  Buffer: {frameBuffer.qsize()}/{frameBuffer.maxsize}     ", end='\r')

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
        
        # Release the video capture and writer
        cap.release()
        if saveVideo and videoWriter is not None:
            videoWriter.release()
            print(f"Video saved successfully.")
        
        # Show playback statistics
        totalTime = time.time() - videoStartTime
        avgFps = frameId / totalTime if totalTime > 0 else 0
        print(f"\nVideo playback complete. Frames: {frameId}, Time: {totalTime:.2f}s, Average FPS: {avgFps:.1f}")
        cv2.destroyAllWindows()

def main():
    # Settings configured directly in the code
        
    # Common settings
    modelPath = r"models\BadAppleModel_48x36_To_480x360_conv\badAppleModel_best.pt"
    videoPath = r"Bad Apple!!.mp4"  # Direct path to the MP4 file

    # Play mode settings
    fps = 30
    displayWidth = DEFAULT_WIDTH
    displayHeight = DEFAULT_HEIGHT
    saveVideo = True  # Set to True to save the output video
     
    playVideo(
        modelPath=modelPath,
        videoPath=videoPath,
        fps=fps,
        displayWidth=displayWidth,
        displayHeight=displayHeight,
        saveVideo=saveVideo
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        cv2.destroyAllWindows()
