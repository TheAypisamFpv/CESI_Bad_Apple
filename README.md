# Bad Apple in Cisco Packet Tracer
Youtube video available here:

[![Bad Apple in Cisco Packet Tracer](https://img.youtube.com/vi/OW7dnr0aOqs/0.jpg)](https://youtu.be/OW7dnr0aOqs?si=Xn7q4KdyrlpGUiIf)


Play Bad Apple in Cisco Packet Tracer using Python, inside a computer in CPT.

## Project Overview

This repository contains a complete pipeline for:
1. Converting video to ASCII art representation
2. Playing the ASCII animation in Cisco Packet Tracer
3. Supporting both PC and router devices in network simulations

## Components

### Video Conversion
- `videoToAscii.py`: Converts the original Bad Apple video into ASCII art, saving it as a text file for playback
- `Bad Apple!!.mp4`: Original video source for conversion (but you can use any video you want !)

### Cisco Packet Tracer Implementation
- `cisco packet tracer/PC/timeAccurateBadApple.py`: Script for playing the ASCII animation on PC devices
- `cisco packet tracer/router/Bad_apple_router.py`: Script for displaying the animation on router devices
- `cisco packet tracer/router/bad_apple_router_config.txt`: Configuration instructions for router display

### Output Files
- `ASCII videos/`: Directory containing generated ASCII text files (e.g., `Bad Apple!!_48x18@30.0fps_6572_0_grayscale.txt`)

## How It Works

1. **Video Processing**: The original video is processed frame by frame and converted to ASCII characters
2. **ASCII Mapping**: Each pixel's brightness is mapped to an appropriate ASCII character to represent different shades
3. **Timing Control**: The script manages playback timing to match the original video's frame rate
4. **Cisco Integration**: Custom scripts enable the ASCII animation to run on network devices within Cisco Packet Tracer

## Usage

### Step 1: Generate ASCII Video
```
python videoToAscii.py
```
Keep default values for the original result.

### Step 2: Prepare the Cisco Packet Tracer Script
1. Open the generated ASCII text file from `ASCII videos/` directory
2. Open `timeAccurateBadApple.py` in a text editor
3. Copy ALL content from the ASCII text file and paste it as the value of the `file` variable (as a String)
4. Save the modified script

### Step 3: Run in Cisco Packet Tracer
1. Open Cisco Packet Tracer
2. Add a PC device to your network topology
3. Create a new Python file in the PC
4. Copy the entire content of your modified `timeAccurateBadApple.py` into this file
5. Run the script in the PC's Python environment

## Technical Details

The ASCII conversion process uses:
- Frame extraction from video
- Resizing to target dimensions (width x height)
- Grayscale conversion for brightness mapping
- Character selection based on pixel intensity
- Frame rate management for accurate playback

Cisco Packet Tracer implementation works around limitations in the network simulator's Python environment by embedding the entire ASCII animation data directly in the script.
