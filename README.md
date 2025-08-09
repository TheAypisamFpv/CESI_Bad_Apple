<div align="center">
  
[![Python](https://img.shields.io/badge/Python-3.6+-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Cisco](https://img.shields.io/badge/Cisco-Packet%20Tracer-1BA0D7?style=for-the-badge&logo=cisco&logoColor=white)](https://www.netacad.com/courses/packet-tracer)
[![ASCII Art](https://img.shields.io/badge/ASCII-Art-blueviolet?style=for-the-badge)](https://en.wikipedia.org/wiki/ASCII_art)
[![Networking](https://img.shields.io/badge/Networking-Simulation-orange?style=for-the-badge&logo=cisco&logoColor=white)](https://www.cisco.com/)

</div>

# 🍎 Bad Apple - Cisco Packet Tracer 🌐

<div align="center">
  
<a href="https://youtu.be/OW7dnr0aOqs?si=Xn7q4KdyrlpGUiIf">
  <img src="https://img.youtube.com/vi/OW7dnr0aOqs/0.jpg" alt="Bad Apple in Cisco Packet Tracer" width="600">
</a>

*▶️ Click the image to watch Bad Apple in Cisco Packet Tracer*

</div>


## 🌟 Project Overview

Play Bad Apple in Cisco Packet Tracer using Python, inside a computer in CPT. This unique implementation brings the iconic animation to network simulation software.

<div align="center">
  <h3>🌐 Network Simulation</h3>
</div>

This repository contains a complete pipeline for:

<table>
  <tr>
    <td>1️⃣</td>
    <td>Converting video to ASCII art representation</td>
  </tr>
  <tr>
    <td>2️⃣</td>
    <td>Playing the ASCII animation in Cisco Packet Tracer</td>
  </tr>
  <tr>
    <td>3️⃣</td>
    <td>Supporting both PC and router devices in network simulations</td>
  </tr>
</table>

## 🧩 Components

<div align="center">
  
### 🎬 Video Conversion
  
</div>

<table>
  <tr>
    <td><code>videoToAscii.py</code></td>
    <td>Converts the original Bad Apple video into ASCII art, saving it as a text file for playback</td>
  </tr>
  <tr>
    <td><code>Bad Apple!!.mp4</code></td>
    <td>Original video source for conversion (but you can use any video you want!)</td>
  </tr>
</table>

<div align="center">
  
### 🖧 Cisco Packet Tracer Implementation
  
</div>

<table>
  <tr>
    <td><code>cisco packet tracer/PC/timeAccurateBadApple.py</code></td>
    <td>Script for playing the ASCII animation on PC devices</td>
  </tr>
  <tr>
    <td><code>cisco packet tracer/router/Bad_apple_router.py</code></td>
    <td>Script for displaying the animation on router devices</td>
  </tr>
  <tr>
    <td><code>cisco packet tracer/router/bad_apple_router_config.txt</code></td>
    <td>Configuration instructions for router display</td>
  </tr>
</table>

<div align="center">
  
### 📂 Output Files
  
</div>

<table>
  <tr>
    <td><code>ASCII videos/</code></td>
    <td>Directory containing generated ASCII text files (e.g., <code>Bad Apple!!_48x18@30.0fps_6572_0_grayscale.txt</code>)</td>
  </tr>
</table>

## ✨ How It Works

<div align="left">
  <img src="https://img.shields.io/badge/1-Video%20Processing-blue?style=for-the-badge" alt="Step 1"/>
</div>

> The original video is processed frame by frame and converted to ASCII characters

<div align="left">
  <img src="https://img.shields.io/badge/2-ASCII%20Mapping-purple?style=for-the-badge" alt="Step 2"/>
</div>

> Each pixel's brightness is mapped to an appropriate ASCII character to represent different shades

<div align="left">
  <img src="https://img.shields.io/badge/3-Timing%20Control-orange?style=for-the-badge" alt="Step 3"/>
</div>

> The script manages playback timing to match the original video's frame rate

<div align="left">
  <img src="https://img.shields.io/badge/4-Cisco%20Integration-darkgreen?style=for-the-badge" alt="Step 4"/>
</div>

> Custom scripts enable the ASCII animation to run on network devices within Cisco Packet Tracer

## 🚀 Usage

<div align="left">
  
### 1️⃣ Generate ASCII Video
  
</div>

```bash
python videoToAscii.py
```

<div align="left">
  <img src="https://img.shields.io/badge/Tip-Keep%20Default%20Values-informational?style=flat-square" alt="Tip"/>
</div>

> Keep default values for the original result.

<div align="left">

<br>

### 2️⃣ Prepare the Cisco Packet Tracer Script
  
</div>

<table>
  <tr>
    <td>📁</td>
    <td>Open the generated ASCII text file from <code>ASCII videos/</code> directory</td>
  </tr>
  <tr>
    <td>📝</td>
    <td>Open <code>timeAccurateBadApple.py</code> in a text editor</td>
  </tr>
  <tr>
    <td>📋</td>
    <td>Copy ALL content from the ASCII text file and paste it as the value of the <code>file</code> variable (as a String)</td>
  </tr>
  <tr>
    <td>💾</td>
    <td>Save the modified script</td>
  </tr>
</table>

<div align="left">

<br>
 
### 3️⃣ Run in Cisco Packet Tracer
  
</div>

<table>
  <tr>
    <td>🖧</td>
    <td>Open Cisco Packet Tracer</td>
  </tr>
  <tr>
    <td>🖥️</td>
    <td>Add a PC device to your network topology</td>
  </tr>
  <tr>
    <td>📄</td>
    <td>Create a new Python file in the PC</td>
  </tr>
  <tr>
    <td>📋</td>
    <td>Copy the entire content of your modified <code>timeAccurateBadApple.py</code> into this file</td>
  </tr>
  <tr>
    <td>▶️</td>
    <td>Run the script in the PC's Python environment</td>
  </tr>
</table>

<br>

## 🧪 Technical Details

<div align="center">
  
### 🛠️ Conversion Process
  
</div>

The ASCII conversion process uses:

<table>
  <tr>
    <td>🎞️</td>
    <td><b>Frame extraction</b> from video</td>
  </tr>
  <tr>
    <td>📏</td>
    <td><b>Resizing</b> to target dimensions (width x height)</td>
  </tr>
  <tr>
    <td>🔍</td>
    <td><b>Grayscale conversion</b> for brightness mapping</td>
  </tr>
  <tr>
    <td>📝</td>
    <td><b>Character selection</b> based on pixel intensity</td>
  </tr>
  <tr>
    <td>⏱️</td>
    <td><b>Frame rate management</b> for accurate playback</td>
  </tr>
</table>

<div align="center">
  
### ⚙️ Implementation Notes
  
</div>

> Cisco Packet Tracer implementation works around limitations in the network simulator's Python environment by embedding the entire ASCII animation data directly in the script.