# 🍎 Bad Apple - Temporal Graph Visualization 📊

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![NetworkX](https://img.shields.io/badge/NetworkX-3.5-orange?style=for-the-badge&logo=graphql&logoColor=white)](https://networkx.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.10.5-11557c?style=for-the-badge)](https://matplotlib.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.12.0-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)

</div>

Convert and visualize the Bad Apple video as a temporal graph using Python. This implementation provides a unique perspective by treating the video as a dynamic network where frames are nodes connected through temporal edges.

## 🌟 Project Overview

This repository implements a graph theory approach to video visualization by:

<table>
  <tr>
    <td>1️⃣</td>
    <td>Converting video frames into graph nodes</td>
  </tr>
  <tr>
    <td>2️⃣</td>
    <td>Creating temporal relationships between frames as edges</td>
  </tr>
  <tr>
    <td>3️⃣</td>
    <td>Visualizing the video as a dynamic graph structure</td>
  </tr>
  <tr>
    <td>4️⃣</td>
    <td>Providing multiple visualization and playback options</td>
  </tr>
</table>

Visualize Bad Apple as a temporal graph using Python, NetworkX, and Matplotlib. This implementation treats the video as a dynamic network where each frame is a node connected through temporal edges. Temporal Graph Visualization

Convert and visualize the Bad Apple video as a temporal graph using Python. This implementation provides a unique perspective by treating the video as a dynamic network where frames are nodes connected through temporal edges.

## 📦 Requirements

<details>
<summary><b>Click to expand package list</b></summary>

```
matplotlib==3.10.5
networkx==3.5
numpy==2.2.6
opencv-python==4.12.0.88
```
</details>

## 🧩 Components

<div align="center">
  
### 🔧 Core Script
  
</div>

<table>
  <tr>
    <td><code>VideoToTemporalGraph.py</code></td>
    <td>Main script that handles video processing and graph visualization</td>
  </tr>
</table>

## ✨ Features

<div align="center">
  
### 🎞️ Frame Extraction
  
</div>

> Converts video into sequential frames with timestamps

<div align="center">
  
### 🕸️ Temporal Graph Creation
  
</div>

Builds a graph where:
<table>
  <tr>
    <td>📊 <b>Nodes</b></td>
    <td>Represent individual frames</td>
  </tr>
  <tr>
    <td>↔️ <b>Edges</b></td>
    <td>Represent temporal transitions between frames</td>
  </tr>
</table>

<div align="center">
  
### 🖥️ Visualization Options
  
</div>

<table>
  <tr>
    <td>🌊 <b>Dynamic Graph Visualization</b></td>
    <td>With sliding window for efficient rendering</td>
  </tr>
  <tr>
    <td>▶️ <b>Frame-by-frame Playback</b></td>
    <td>For detailed analysis</td>
  </tr>
  <tr>
    <td>📈 <b>Performance Monitoring</b></td>
    <td>Frame drop statistics and rendering metrics</td>
  </tr>
</table>

## 🚀 Usage

<div align="center">
  
### 1️⃣ Set up the Python environment
  
</div>

```bash
pip install -r requirements.txt
```

<div align="center">
  
### 2️⃣ Run the script
  
</div>

```bash
python VideoToTemporalGraph.py
```

<div align="center">
  
### 3️⃣ Follow the prompts
  
</div>

<table>
  <tr>
    <td>📁</td>
    <td>Input video file path</td>
  </tr>
  <tr>
    <td>⏳</td>
    <td>Wait for the <code>.pkl</code> file to be created</td>
  </tr>
  <tr>
    <td>📂</td>
    <td>Input the <code>.pkl</code> file path to load and visualize the temporal graph</td>
  </tr>
</table>

## 🧪 Technical Details

<div align="center">
  
### 🛠️ Technologies Used
  
</div>

<table>
  <tr>
    <td><img src="https://img.shields.io/badge/NetworkX-orange?style=flat-square" alt="NetworkX"/></td>
    <td>Graph structure and manipulation</td>
  </tr>
  <tr>
    <td><img src="https://img.shields.io/badge/OpenCV-blue?style=flat-square" alt="OpenCV"/></td>
    <td>Video frame extraction</td>
  </tr>
  <tr>
    <td><img src="https://img.shields.io/badge/Matplotlib-green?style=flat-square" alt="Matplotlib"/></td>
    <td>Visualization and animation</td>
  </tr>
  <tr>
    <td><img src="https://img.shields.io/badge/Pickle-purple?style=flat-square" alt="Pickle"/></td>
    <td>Saving/loading graph data</td>
  </tr>
</table>