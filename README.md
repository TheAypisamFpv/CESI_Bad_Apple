# Bad Apple - Temporal Graph Visualization
Youtube video available here:

[![Bad Apple!! but with Temporal Graph](https://img.youtube.com/vi/XJBJw7phDGY/0.jpg)](https://youtu.be/XJBJw7phDGY?si=dxtRH8JJp5Airdj6)

Convert and visualize the Bad Apple video as a temporal graph using Python. This implementation provides a unique perspective by treating the video as a dynamic network where frames are nodes connected through temporal edges.

## Project Overview

This repository implements a graph theory approach to video visualization by:
1. Converting video frames into graph nodes
2. Creating temporal relationships between frames as edges
3. Visualizing the video as a dynamic graph structure
4. Providing multiple visualization and playback options

Visualize Bad Apple as a temporal graph using Python, NetworkX, and Matplotlib. This implementation treats the video as a dynamic graph where each frame is a node connected through temporal edges.

### Requirements
```
matplotlib==3.10.5
networkx==3.5
numpy==2.2.6
opencv-python==4.12.0.88
```

### Components

- `VideoToTemporalGraph.py`: Main script that handles video processing and graph visualization

### Features

1. **Frame Extraction**: Converts video into sequential frames with timestamps
2. **Temporal Graph Creation**: Builds a graph where:
   - Nodes represent individual frames
   - Edges represent temporal transitions between frames
3. **Visualization Options**:
   - Dynamic graph visualization with sliding window
   - Frame-by-frame playback
   - Performance monitoring (frame drop statistics)

### Usage

1. Set up the Python environment:
   ```
   pip install -r requirements.txt
   ```

2. Run the script:
   ```
   python VideoToTemporalGraph.py
   ```

3. Follow the prompts to:
   - Input video file path
   - Wait for the `.pkl` file to be created
   - Input the `.pkl` file path to load and visualize the temporal graph

### Technical Details

The implementation uses:
- NetworkX for graph structure and manipulation
- OpenCV for video frame extraction
- Matplotlib for visualization and animation
- Pickle for saving/loading graph data

The temporal graph structure provides a unique way to analyze the video as a dynamic network, with features like:
- Frame transition analysis
- Temporal relationship visualization
- Performance optimization through frame skipping
- Memory-efficient sliding window display
