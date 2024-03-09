import time



file_path = input("Enter the name of the file you want to convert (must be in 'ASCII videos' folder):\n> ")
with open("ASCII videos\\" + file_path, "r") as f:
    file = f.read()

frame_nbr = int(file_path.split("_")[-3])
properties = file_path.split("_")[-4]

color = file_path.split("_")[-1].removesuffix(".txt") == "colored"

size = properties.split("@")[0]
frame_height = int(size.split("x")[1])
frame_width = int(size.split("x")[0])

fps = float(properties.split("@")[1].removesuffix("fps"))

print(f"please make sure all gathered information is correct:\n\tframe_nbr = {frame_nbr}\n\tframe_width = {frame_width}\n\tframe_height = {frame_height}\n\tfps = {fps}\n\tcolor = {color}\n")
input("Press enter to continue...")


playback_fps = fps #this will not alter the video lenght, but only render at that frame rate
sleep_time = 1/(playback_fps*2)
to_sleep = fps != playback_fps


frame_char_nbr = frame_width * frame_height


print("\n")

i = 0
prev_i = 0
start_time = time.time()
dropped_frames = 0
while i < frame_nbr:
    i = round((time.time() - start_time)*fps)
    if i != prev_i:
        dropped_frames += i - (prev_i + 1)

        pre_frame = file[i * frame_char_nbr:(i + 1) * frame_char_nbr]
        # Use a generator expression instead of a list comprehension
        frame_lines = (pre_frame[h * frame_width:(h + 1) * frame_width] for h in range(frame_height))
        frame = "\n".join([str(i)] + list(frame_lines))

        #clear output
        print("\033[H\033[J",frame)
        prev_i = i
        if to_sleep: time.sleep(sleep_time)

 
print("Duration : {}s".format(round(time.time() - start_time)))
dropped_frame_percent = round((dropped_frames*100)/frame_nbr)
excepted_dropped_frame_percent = round((1-(playback_fps/fps))*100)

print("Dropped frames : {} ({}%, excepted dropped frames percentage is {}%)".format(dropped_frames, dropped_frame_percent, excepted_dropped_frame_percent))
