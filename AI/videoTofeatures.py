import os
import cv2
import pandas as pd
import hashlib

def videoToCsv(videoPath:str, outputCsv:str, initialScalingFactor:float=1.0, ouputScalingFactor:float=0.5, frameSkip=1):
    """
    Convert a video to a CSV file where each row is a flattened frame.

    csv structure:
    ```csv
        resizedFrameData, originalFrameData
        [pixelValue1, pixelValue2, ...], [originalPixelValue1, originalPixelValue2, ...]
        [pixelValue1, pixelValue2, ...], [originalPixelValue1, originalPixelValue2, ...]
        ...
    ```
    ---
    Args:
        - videoPath (str): Path to the video file.
        - outputCsv (str): Path to the output CSV file.
        - initialScalingFactor (float): Scaling factor to apply to the video frames before converting them to CSV.
        - ouputScalingFactor (float): Scaling factor to apply to the video frames after converting them to CSV.
        - frameSkip (int): Number of frames to skip when converting the video to CSV.
    """
    print("Converting video to CSV...")

    print("\tLoading video...", end=" ")
    cap = cv2.VideoCapture(videoPath)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    print("done.")

    # print the original video's properties and the new video's properties
    print("\tInitial video properties:")
    initialFps = cap.get(cv2.CAP_PROP_FPS)
    initialFrameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    initialFrameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    initialFrameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"\t  - FPS: {initialFps}")
    print(f"\t  - Frame count: {initialFrameCount}")
    print(f"\t  - Frame width: {initialFrameWidth}")
    print(f"\t  - Frame height: {initialFrameHeight}")

    print(f"\n\tNew video properties: (scaling factor: {initialScalingFactor}, frame skip: {frameSkip})")
    newFps = initialFps
    newFrameCount = initialFrameCount // frameSkip
    initialNewFrameWidth = int(initialFrameWidth * initialScalingFactor)
    initialNewFrameHeight = int(initialFrameHeight * initialScalingFactor)

    outputNewFrameWidth = int(initialFrameWidth * ouputScalingFactor)
    outputNewFrameHeight = int(initialFrameHeight * ouputScalingFactor)
    
    print(f"\t  - FPS: {newFps}")
    print(f"\t  - Frame count: {newFrameCount}")
    print(f"\t  - Frame width: input:{initialNewFrameWidth} ouput:{outputNewFrameWidth}")
    print(f"\t  - Frame height: input:{initialNewFrameHeight} ouput:{outputNewFrameHeight}")


    frameId = 0
    framesData = []

    print()

    i = 0
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break
        
        # Only process every frameSkip-th frame
        if frameId % frameSkip != 0:
            frameId += 1
            continue

        i += 1
        print(f"\tProcessing frame {i}/{newFrameCount}...", end="\r")

        # Convert the frame to grayscale
        originalFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        

        # Resize the frame
        inputFrame = cv2.resize(originalFrame, (initialNewFrameWidth, initialNewFrameHeight))
        outputFrame = cv2.resize(originalFrame, (outputNewFrameWidth, outputNewFrameHeight))

        # resize = 10
        # cv2.imshow("Original", originalFrame)
        # cv2.imshow("Input", inputFrame)
        # cv2.imshow("Output", outputFrame)
        # cv2.waitKey(20)

        # Normalize the frames
        inputFrame = inputFrame / 255.0
        outputFrame = outputFrame / 255.0
    
        # Flatten the frames
        inputFrameData = inputFrame.flatten().tolist()
        outputFrameData = outputFrame.flatten().tolist()

        # inputFrameData = [1 if x > 0.5 else 0 for x in inputFrameData]
        # outputFrameData = [1 if x > 0.5 else 0 for x in outputFrameData]
        
        framesData.append([inputFrameData, outputFrameData])

        frameId += 1

    print(f"\tProcessing frame done.         ")

    outputCsvName = outputCsv.removesuffix('.csv')

    # create a hash of the csv file using the frames data
    print("\n\tHashing the CSV file...", end=" ")
    csvHash = hashlib.md5(str(framesData).encode()).hexdigest()
    print("done.")


    # save input and output frame size to csv
    print("\n\tSaving frame size to CSV...", end=" ")
    frameDimensions = pd.DataFrame([[initialNewFrameWidth, initialNewFrameHeight, outputNewFrameWidth, outputNewFrameHeight, csvHash]], columns=['initialNewFrameWidth', 'initialNewFrameHeight', 'outputNewFrameWidth', 'outputNewFrameHeight', 'hash'])
    frameDimensions.to_csv(outputCsv.removesuffix('.csv') + '_size.csv', index=False)
    print("done.")


    print(f"\tSaving CSV `{outputCsv}`...", end=" ")

    outputCsv = f"{outputCsvName}_{initialNewFrameWidth}x{initialNewFrameHeight}.csv"
    
    videoFrameDataset = pd.DataFrame(framesData, columns=['resizedFrameData', 'originalFrameData'])    
    videoFrameDataset.to_csv(outputCsv, index=False)
    print("done.")

    print()


if __name__ == '__main__':
    rootDir = os.path.dirname(os.path.abspath(__file__))
    videoPath = 'D:/VS_Python_Project/CESI_Bad_Apple/Bad Apple!!.mp4'
    initialScalingFactor = 0.05
    ouputScalingFactor = 0.25
    frameSkip = 1
    outputCsv = os.path.join(rootDir, 'Bad Apple!!.csv')
    videoToCsv(videoPath=videoPath, outputCsv=outputCsv, initialScalingFactor=initialScalingFactor, ouputScalingFactor=ouputScalingFactor, frameSkip=frameSkip)