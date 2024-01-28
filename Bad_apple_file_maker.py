import time

with open("bad_apple_Original.txt", 'r') as f:
    file = f.read()


FRAME_NBR = 6571
"""original frame number"""

FRAME_WIDTH = 150
"""original frame width"""

FRAME_HEIGHT = 45
"""original frame height"""

FRAME_CHAR_NBR = FRAME_WIDTH * FRAME_HEIGHT
"""original number of characters in a frame"""

# ----PARAMETERS----
FRAME_SKIP = 2
"""use 1 frame every <FRAME_SKIP>"""

RESIZE_FACTOR = 0.5
"""resize the frame by a factor of <RESIZE_FACTOR>"""

file_name = f"bad_apple_frames_{RESIZE_FACTOR}x_{int(30/FRAME_SKIP)}fps.txt"
final_file_size = 0
"""final file size in KiloBytes"""


print("\n")
print("Starting...")
print(f"Creating Bad Apple!! file frames {RESIZE_FACTOR}x the original size at {int(30/FRAME_SKIP)}fps")

with open(file_name, 'w') as f:
    for i in range(0, FRAME_NBR, FRAME_SKIP):
        frame = ""
        pre_frame = file[i*FRAME_CHAR_NBR:(i+1)*FRAME_CHAR_NBR]

        for h in range(1, FRAME_HEIGHT, int(1/RESIZE_FACTOR)):
            try:
                line = pre_frame[h*FRAME_WIDTH:(h+1)*FRAME_WIDTH]
                for w in range(0, FRAME_WIDTH, int(1/RESIZE_FACTOR)):
                    try:
                        frame += line[w]
                    except:
                        continue
            except:
                continue

        f.write(frame)


print("Finished")
with open(file_name, 'r') as f:
    final_file_size = len(f.read()) / 1000

print(f"Final file '{file_name}' size: {final_file_size}KB")