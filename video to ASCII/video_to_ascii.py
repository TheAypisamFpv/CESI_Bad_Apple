import cv2, os, tqdm
import numpy as np


#-------PARAMETERS-------#
default = '\033['
reset = '\033[0m'

colors = {
    "30": [0,0,0],
    "31": [128,0,0],
    "32": [0,128,0],
    "33": [128,128,0],
    "34": [0,0,128],
    "35": [128,0,128],
    "36": [0,128,128],
    "37": [192,192,192],
    }

# print the color table
print("Color table :")
for color in colors:
    print(f"{default}{color}m{color} : {colors[color]}")

VIDEO_PATH = "input video\\"
VIDEO_PATH += input("Enter the name of the video you want to convert (must be in 'input video' folder):\n> ")
if not os.path.isfile(VIDEO_PATH):
    raise Exception("The video file doesn't exist")

COLOR = False if input("Convert to grayscale Y/n :\n> ") == "" else True




def find_color(pixel:list[int]):
    closest = ""
    closest_dist = 255*3
    for color in colors:
        dist = 0
        for i in range(3):
            dist += abs(pixel[i] - colors[color][i])
        if dist < closest_dist:
            closest_dist = dist
            closest = color
    return closest

inputs = input("Resize factor (default 10) (ASCII video will have <RESIZE_FACTOR> times smaller dimension):\n> ")
RESIZE_FACTOR = 10 if inputs == "" else int(inputs)
"""Determines the number of pixels par character
---
default 25 : 5*5px to 1chr

"""

CHARS = np.asarray(list(' ._,:;-÷+=*?%S#@'))
""" ._,:;-÷+=*?%S#@"""

INVERTED = False if input("Invert the video (default white = char, black = no char) (default False) y/n(leave empty) :\n> ") == "" else True

if INVERTED:
    CHARS = CHARS[::-1]

inputs = input("Skip frames (default 1) :\n> ")    
SKIP_FRAMES = 1 if inputs == "" else round(inputs)
"""Use 1 frame every <SKIP_FRAMES>"""

inputs = input("Storage method 0:one line  |  1:frame per frame (default 0) :\n> ")
STORAGE_METHOD = 0 if inputs == "" else round(inputs)
"""Determines how the frames are stored in the final file
---
0 : All the frames are stored on one line
1 : Each frame is stored on a new line"""

#------------------------#
#convert all the frames of the video to ASCII
cap = cv2.VideoCapture(VIDEO_PATH)
ascii_frames = []
i = 0

width, height, fps = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)//RESIZE_FACTOR), round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)//(RESIZE_FACTOR*2)), cap.get(cv2.CAP_PROP_FPS)

print(f"Converting '{VIDEO_PATH}' to ASCII...")
frame_nbr = 0
nbr_frames = round(cap.get(cv2.CAP_PROP_FRAME_COUNT))
pgbar = tqdm.tqdm(total=nbr_frames)

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break

    if i % SKIP_FRAMES == 0:
        if not COLOR:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        frame = cv2.resize(frame, (round(frame.shape[1]//RESIZE_FACTOR), round(frame.shape[0]//(RESIZE_FACTOR*2))), interpolation=cv2.INTER_AREA)

        frame_nbr += 1
        
        # convert the frame to ASCII
        for row in frame:
            for pixel in row:
                if COLOR:
                    rgb_pixel = find_color(pixel)
                    ascii_frames.append(f'\033[{rgb_pixel}m')
                    #confert the pixel to WB for luminosity conversion
                    pixel = (pixel[0] + pixel[1] + pixel[2]) / 3
                    
                #convert the luminosity of the pixel to ASCII
                ascii_frames.append(CHARS[round(pixel/255*(len(CHARS)-1))])
                
            if STORAGE_METHOD == 1:
                ascii_frames.append('\n')
        if STORAGE_METHOD == 1:
            ascii_frames.append('\n')
    i += 1
    pgbar.update(1)

pgbar.close()

file_type = "colored" if COLOR else "grayscale"
if not COLOR:
    file_type += "-inverted" if INVERTED else ""

file_name = "ASCII videos\\" + VIDEO_PATH.split("\\")[-1].split(".")[0] + f"_{width}x{height}@{fps}fps_{frame_nbr}_{STORAGE_METHOD}_{file_type}.txt"
#save the ASCII frames in a file
with open(file_name, "w") as f:
    f.write(''.join(ascii_frames))
    
print(f"ASCII video saved in '{file_name}'")
#print the ascii video dimensions
print(f"ASCII video settings : {width}x{height}@{fps}fps_{frame_nbr}_{STORAGE_METHOD}")