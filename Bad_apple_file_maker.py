with open("bad_apple_Original.txt", 'r') as f:
    file = f.read()


# ----PARAMETERS----
FRAME_SKIP = 1
"""use 1 frame every <FRAME_SKIP>"""

RESIZE_FACTOR = 0.25
"""resize the frame by a factor of <RESIZE_FACTOR>"""


FRAME_NBR = 6571
"""original frame number"""

FRAME_WIDTH = 150
"""original frame width"""

FRAME_HEIGHT = 45
"""original frame height"""

FRAME_CHAR_NBR = FRAME_WIDTH * FRAME_HEIGHT
"""original number of characters in a frame"""


file_name = f"bad_apple_frames_{RESIZE_FACTOR}x_{round(30/FRAME_SKIP)}fps.txt"
final_file_size = 0
"""final file size in KiloBytes"""

final_frame_width = 0
final_frame_height = 0

print("\n")
print("Starting...")
print(f"Creating Bad Apple!! file frames {RESIZE_FACTOR}x the original size at {round(30/FRAME_SKIP)}fps")

with open(file_name, 'w') as f:
    frame_nbr = 0
    for i in range(0, FRAME_NBR, FRAME_SKIP):
        frame_nbr += 1
        frame = ""
        pre_frame = file[i*FRAME_CHAR_NBR:(i+1)*FRAME_CHAR_NBR]
        H = 0
        for h in range(1, FRAME_HEIGHT, round(1/RESIZE_FACTOR)):
            try:
                line = pre_frame[h*FRAME_WIDTH:(h+1)*FRAME_WIDTH]
                H += 1
                final_frame_height = max(final_frame_height, H)
                W = 0
                for w in range(0, FRAME_WIDTH, round(1/RESIZE_FACTOR)):
                    try:
                        frame += line[w]
                        W += 1
                        final_frame_width = max(final_frame_width, W)
                    except:
                        continue
            except:
                continue

        f.write(frame)


print("Finished")
with open(file_name, 'r') as f:
    final_file_size = len(f.read()) / 1000

print(f"Final file '{file_name}' size: {final_file_size}KB")

print(f"Final frame dimensions: {final_frame_width}x{final_frame_height} with {frame_nbr} frames")