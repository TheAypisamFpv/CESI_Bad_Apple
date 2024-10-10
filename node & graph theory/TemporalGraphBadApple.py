import cv2
import time
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle

SavedGraphPath = 'BadApple.pkl'

# Step 1: Extract Frames from the Video with Timestamps
def ExtractFramesWithTimestamps(video_path):
    """
    Extracts frames from a video file along with their timestamps.
    """
    print("Extracting frames and timestamps from the video...")
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # Get the video's frame rate
    Frames = []
    TimeStamps = []

    StartTime = time.time()  # Record the start time
    while cap.isOpened():
        ret, Frame = cap.read()
        if not ret:
            break
        GrayFrame = cv2.cvtColor(Frame, cv2.COLOR_BGR2GRAY)
        Frames.append(GrayFrame)
        
        # Record the timestamp for the current frame
        ElapsedTime = time.time() - StartTime
        TimeStamps.append(ElapsedTime)

    cap.release()
    return Frames, TimeStamps, fps

# Step 2: Create a Temporal Graph from the Frames
def CreateTemporalGraph(Frames, TimeStamps):
    """
    Creates a temporal graph from the frames and timestamps.
    """
    print("Creating a temporal graph from the frames...")
    
    TemporalGraph = nx.Graph()

    for i in range(len(Frames)):
        TemporalGraph.add_node(i, frame=Frames[i], timestamp=TimeStamps[i])

    # Add edges between consecutive frames
    for i in range(len(Frames) - 1):
        TemporalGraph.add_edge(i, i + 1, time_interval=TimeStamps[i + 1] - TimeStamps[i])

    return TemporalGraph

# Step 3: Save the Temporal Graph and Frames
def SaveTemporalGraph(TemporalGraph, Frames, Filename=SavedGraphPath):
    """
    Saves the temporal graph and frames to a file using pickle.
    """
    print("Saving the temporal graph and frames to a file...")
    
    with open(Filename, 'wb') as f:
        # Save the graph and frames as a dictionary
        pickle.dump({'graph': TemporalGraph, 'frames': Frames}, f)

# Step 4: Load the Temporal Graph and Frames
def LoadTemporalGraph(Filename=SavedGraphPath):
    """
    Loads the temporal graph and frames from the saved file.
    """
    print("Loading the temporal graph and frames from the file...")
    
    with open(Filename, 'rb') as File:
        # Load the graph and frames from the saved dictionary
        Data = pickle.load(File)
        TemporalGraph = Data['graph']
        Frames = Data['frames']
    return TemporalGraph, Frames

# Step 5: Visualize the Temporal Graph
import numpy as np

def VisualizeTemporalGraph(TemporalGraph):
    """
    Visualizes the temporal graph in 3D space with frames as nodes and transitions as edges.
    """
    print("Visualizing the temporal graph...")
    
    # Create a 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    print("Positioning nodes using a spring layout...")
    pos = nx.spring_layout(TemporalGraph, dim=3)  # Position nodes in 3D space

    # Extract the x, y, and z coordinates from the positions
    x_vals = [pos[node][0] for node in TemporalGraph.nodes]
    y_vals = [pos[node][1] for node in TemporalGraph.nodes]
    z_vals = [pos[node][2] for node in TemporalGraph.nodes]  # Can represent timestamp or frame index

    # Draw the nodes
    ax.scatter(x_vals, y_vals, z_vals, c='lightblue', s=700, label='Frames')  # Nodes

    # Draw edges
    for edge in TemporalGraph.edges:
        print(f"Drawing edge between nodes {edge[0]} and {edge[1]}...", end='\r')
        x_start, y_start, z_start = pos[edge[0]]
        x_end, y_end, z_end = pos[edge[1]]
        ax.plot([x_start, x_end], [y_start, y_end], [z_start, z_end], color='gray')  # Edges

    # Draw frame images on top of the nodes
    for node in TemporalGraph.nodes:
        print(f"Drawing frame at node {node}...                                     ", end='\r')
        x, y, z = pos[node]
        frame = TemporalGraph.nodes[node]['frame']

        # Convert the frame to RGB format if it is grayscale
        if len(frame.shape) == 2:  # Grayscale frame
            frame_rgb = np.stack((frame,) * 3, axis=-1)  # Convert to RGB by stacking

        # Use plot_surface to display the frame as a 2D image at the (x, y, z) position
        img_extent = (x - 0.1, x + 0.1, y - 0.1, y + 0.1)  # Define the extent for the image
        ax.plot_surface(np.array([[x - 0.1, x + 0.1], [x - 0.1, x + 0.1]]),
                        np.array([[y - 0.1, y - 0.1], [y + 0.1, y + 0.1]]),
                        np.array([[z, z], [z, z]]) + 0.1,  # Slightly above the nodes
                        facecolors=frame_rgb / 255,  # Normalize to [0, 1] for color
                        rstride=100, cstride=100, alpha=0.8)

    ax.set_title('Temporal Graph of Video Frames in 3D')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Time / Frame Index')  # You can modify this label based on what Z represents
    plt.legend()
    plt.show()



# Step 6: Play the Video Using the Temporal Graph
def PlayVideoWithTemporalGraph(TemporalGraph, fps):
    """
    Plays the video by traversing the temporal graph and displaying frames.
    """
    print("Playing the video using the temporal graph...")
    
    CurrentNode = 0
    StartTime = time.time()  # Track overall start time
    
    while CurrentNode in TemporalGraph.nodes:
        CurrentTime = time.time()
        ElapsedTime = CurrentTime - StartTime
        
        # Check if enough time has passed to show the next frame
        if ElapsedTime >= TemporalGraph.nodes[CurrentNode]['timestamp']:
            frame = TemporalGraph.nodes[CurrentNode]['frame']
            plt.imshow(frame, cmap='gray')
            plt.axis('off')
            plt.show(block=False)
            plt.pause(0.001)  # Short pause to allow frame display
            plt.clf()
            CurrentNode += 1  # Move to the next node

# Main function to execute the steps
def main():
    VideoPath = 'Bad Apple!!.mp4'  # Change to the actual video file path

    # Step 1: Extract frames and timestamps from the video
    Frames, TimeStamps, fps = ExtractFramesWithTimestamps(VideoPath)

    # Step 2: Create a temporal graph where nodes are frames and edges represent transitions
    TemporalGraph = CreateTemporalGraph(Frames, TimeStamps)

    # Step 3: Save the temporal graph and frames to a file
    SaveTemporalGraph(TemporalGraph, Frames)

    # Optionally, load the temporal graph from a file (comment this out if not needed)
    LoadedTemporalGraph, LoadedFrames = LoadTemporalGraph()

    # Step 4: Visualize the temporal graph
    # VisualizeTemporalGraph(LoadedTemporalGraph)

    # Step 5: Play the video by traversing the temporal graph
    PlayVideoWithTemporalGraph(LoadedTemporalGraph, fps)

# Run the main function
if __name__ == "__main__":
    LoadedTemporalGraph, LoadedFrames = LoadTemporalGraph()

    # VisualizeTemporalGraph(LoadedTemporalGraph)

    PlayVideoWithTemporalGraph(LoadedTemporalGraph, 30)
