import cv2, os
import numpy as np


#-------PARAMETERS-------#


VIDEO_PATH = "input video\\"
VIDEO_PATH += input("Enter the name of the video you want to convert (must be in 'input video' folder):\n> ")
if not os.path.isfile(VIDEO_PATH):
    raise Exception("The video file doesn't exist")

inputs = input("Resize factor (default 25) :\n> ")
RESIZE_FACTOR = 25 if inputs == "" else int(inputs)
"""Determines the number of pixels par character
---
default 25 : 5*5px to 1chr
...
"""

CHARS = np.asarray(list(' .,:;+*?%S&$#@'))
""" .,:;+*?%S&$#@"""

INVERTED = False if input("Invert the video (default white = char, black = no char) (default False) y/n(leave empty) :\n> ") == "" else True

if INVERTED:
    CHARS = CHARS[::-1]

inputs = input("Skip frames (default 1) :\n> ")    
SKIP_FRAMES = 1 if inputs == "" else int(inputs)
"""Use 1 frame every <SKIP_FRAMES>"""

inputs = input("Storage method 0:one line  |  1:frame per frame (default 0) :\n> ")
STORAGE_METHOD = 0 if inputs == "" else int(inputs)
"""Determines how the frames are stored in the final file
---
0 : All the frames are stored on one line
1 : Each frame is stored on a new line"""

#------------------------#
#convert all the frames of the video to ASCII
cap = cv2.VideoCapture(VIDEO_PATH)
ascii_frames = []
i = 0

width, height, fps = 0,0,0

print(f"Converting '{VIDEO_PATH}' to ASCII...")
frame_nbr = 0
while cap.isOpened():
    ret, frame = cap.read()
    nbr_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not ret:
        break

    
    print(f"{int(i/nbr_frames*100):3d}%", end="\r")
    if i % SKIP_FRAMES == 0:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, (frame.shape[1]//RESIZE_FACTOR, frame.shape[0]//(RESIZE_FACTOR*2)), interpolation=cv2.INTER_AREA)

        width, height, fps, frame_nbr = frame.shape[1], frame.shape[0], cap.get(cv2.CAP_PROP_FPS), frame_nbr +1
        
        # convert the frame to ASCII
        for row in frame:
            for pixel in row:
                #convert the luminosity of the pixel to ASCII
                ascii_frames.append(CHARS[int(pixel/255*(len(CHARS)-1))])
            if STORAGE_METHOD == 1:
                ascii_frames.append("\n")
        if STORAGE_METHOD == 1:
            ascii_frames.append("\n")
    i += 1

file_name = "ASCII videos\\" + VIDEO_PATH.split("\\")[-1].split(".")[0] + f"_{width}x{height}@{fps}fps_{frame_nbr}_{STORAGE_METHOD}_{'normal' if not INVERTED else 'inverted'}.txt"
#save the ASCII frames in a file
with open(file_name, "w") as f:
    f.write("".join(ascii_frames))
    
print(f"ASCII video saved in '{file_name}'")
#print the ascii video dimensions
print(f"ASCII video settings : {width}x{height}@{fps}fps_{frame_nbr}_{STORAGE_METHOD}")