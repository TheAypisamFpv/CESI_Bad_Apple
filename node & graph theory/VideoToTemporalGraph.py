import cv2
import time
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle
import os
from networkx import Graph


def ExtractFramesWithTimestamps(VideoPath=''):
    """
    Extracts frames and timestamps from a video file.
    """
    print()
    
    if VideoPath == "":
        VideoPath = input("Enter the path to the video file you want to extract frames and timestamps from (with extension):\n> ")

    if not os.path.exists(VideoPath):
        raise FileNotFoundError(f"Video file not found at: {VideoPath}")
    
    print("Extracting frames and timestamps from the video...")
    
    cap = cv2.VideoCapture(VideoPath)
    fps = float(cap.get(cv2.CAP_PROP_FPS))  # Get the video's frame rate
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


    
    return Frames, TimeStamps, fps, os.path.basename(VideoPath).split(".")[0]


def CreateTemporalGraph(Frames:list, TimeStamps:list):
    """
    Creates a temporal graph from the frames and timestamps.
    """
    print("\nCreating a temporal graph from the frames...")
    
    TemporalGraph = nx.Graph()

    for i in range(len(Frames)):
        TemporalGraph.add_node(i, frame=Frames[i], timestamp=TimeStamps[i])

    # Add edges between consecutive frames
    for i in range(len(Frames) - 1):
        TemporalGraph.add_edge(i, i + 1, time_interval=TimeStamps[i + 1] - TimeStamps[i])

    return TemporalGraph


def SaveTemporalGraph(TemporalGraph:Graph, Frames:list, fps:float, Filename=''):
    """
    Saves the temporal graph and frames to a file.
    Returns the filename where the data is saved.
    """
    print()
    
    if Filename == "":
        Filename = input("Enter the name of the file you want to save the temporal graph and frames to (without extension):\n> ")
    
    Filename = f"{Filename}_{fps}.pkl".replace(" ", "-")

    print(f"Saving the temporal graph and frames to the file: {Filename}...")
    
    with open(Filename, 'wb') as f:
        # Save the graph and frames as a dictionary
        pickle.dump({'graph': TemporalGraph, 'frames': Frames}, f)

    return Filename


def LoadTemporalGraph(Filename=""):
    """
    Loads the temporal graph and frames from a file.
    """
    if Filename == "":
        Filename = input("\nEnter the name of the file you want to load the temporal graph and frames from (with extension):\n> ")

    print(f"\nLoading the temporal graph and frames from the file: {Filename}...")

    fps = float(Filename.split("_")[-1].removesuffix(".pkl"))
    
    with open(Filename, 'rb') as File:
        # Load the graph and frames from the saved dictionary
        Data = pickle.load(File)
        TemporalGraph = Data['graph']
        Frames = Data['frames']

    print(f"please make sure all gathered information is correct:\n\tframe_nbr = {len(Frames)}\n\tfps = {fps}\n")
    input("Press enter to continue...")
        
    return Graph(TemporalGraph), list(Frames), fps


def VisualizeTemporalGraph(TemporalGraph:Graph, fps:float):
    """
    Visualizes the temporal graph in 2D space with frames as nodes and transitions as edges.
    Only the current frame, the previous frame (n-1), and the next frame (n+1) are rendered at any time.
    The view scrolls to the right at the specified frame rate (fps), adjusting for any lag.
    """
    print("\nVisualizing the temporal graph in 2D...")

    # Create a 2D plot
    fig, ax = plt.subplots(figsize=(24, 16))

    # Manually position nodes in a line along the x-axis
    pos = {node: (node * 5, 0) for node in TemporalGraph.nodes}  # x increases for each node, y = 0

    # Draw edges (these are static and drawn once)
    for edge in TemporalGraph.edges:
        x_start, y_start = pos[edge[0]]
        x_end, y_end = pos[edge[1]]
        ax.plot([x_start, x_end], [y_start, y_end], color='gray')  # Edges

    print("Drawing edges done!")

    # Placeholder for images (we will only show three at a time)
    img_plots = {}

    # Function to draw frame images at nodes
    def DrawFrameAtNode(node):
        if node in img_plots:
            return  # Skip if already drawn

        x, y = pos[node]
        frame = TemporalGraph.nodes[node]['frame']

        # Downsample the frame to reduce the size
        frame_rgb = frame[::5, ::5]  # Reduce size by a factor of 5 (can be adjusted)

        # Define the extent for the frame image
        img_extent = (x - 2, x + 2, y - 2, y + 2)

        # Display the frame as an image and keep reference to remove later
        img_plot = ax.imshow(frame_rgb, cmap='gray', extent=img_extent, alpha=1, aspect='auto')
        img_plots[node] = img_plot

    # Function to clear the images for nodes no longer in view
    def ClearFrameAtNode(node):
        if node in img_plots:
            img_plots[node].remove()
            del img_plots[node]

    # Animation function to update the view window and images
    def update(n):
        current_view_start = n * 5  # Move 5 units per frame
        ax.set_xlim(current_view_start - 5, current_view_start + 5)  # Update the x-axis view window

        # Clear frames no longer in view (n-2 and n+2 are too far)
        if n > 1:
            ClearFrameAtNode(n - 2)
        if n < len(TemporalGraph.nodes) - 2:
            ClearFrameAtNode(n + 2)

        # Draw only frame n, n-1, and n+1
        if n > 0:
            DrawFrameAtNode(n - 1)
        DrawFrameAtNode(n)
        if n < len(TemporalGraph.nodes) - 1:
            DrawFrameAtNode(n + 1)

    # Set the limits for y-axis
    ax.set_ylim(-5, 5)  # Keep y-axis small for visibility
    ax.set_title('Temporal Graph of Video Frames in 2D')
    

    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=len(TemporalGraph.nodes), interval=int(1000/fps), repeat=False)

    plt.legend()
    plt.show()


def PlayVideoWithTemporalGraph(TemporalGraph:Graph, fps:float):
    """
    Plays the video by traversing the temporal graph and displaying frames.
    """
    print("\nPlaying the video using the temporal graph...")

    plt.figure(num='Bad Apple!! with Temporal Graph')

    CurrentNode = 0
    SkipedFrames = 0
    StartTime = time.time() + 0.1  # Start time with a small delay to avoid skipping the first frame due to plt window creation time
    while CurrentNode in TemporalGraph.nodes:
        # Calculate the expected frame index based on elapsed time and fps
        ExpectedFrameIndex = round((time.time() - StartTime) * fps)
    
        # Skip frames to catch up to the expected frame index
        if CurrentNode < ExpectedFrameIndex:
            SkipedFrames += (ExpectedFrameIndex - CurrentNode)
            CurrentNode = ExpectedFrameIndex
    
        frame = TemporalGraph.nodes[CurrentNode]['frame']
        plt.imshow(frame, cmap='gray')
        plt.axis('off')
        plt.show(block=False)
        plt.pause(0.0005)  # Short pause to allow frame display
        plt.clf()
        CurrentNode += 1  # Move to the next node

        # if CurrentNode % 50 == 0:
        #     print(f"Frames drop: {SkipedFrames}/{CurrentNode} ({((SkipedFrames*100)/CurrentNode):.2f}%)", end='\r')

    print(f"Frames drop: {SkipedFrames}/{CurrentNode} ({((SkipedFrames*100)/CurrentNode):.2f}%)", end='\n\n')



def main():
    # Extract frames and timestamps from the video
    Frames, TimeStamps, fps, filename = ExtractFramesWithTimestamps()

    # Create a temporal graph where nodes are frames and edges represent transitions
    TemporalGraph = CreateTemporalGraph(Frames=Frames, TimeStamps=TimeStamps)

    # Save the temporal graph and frames to a file
    savedFilename = SaveTemporalGraph(TemporalGraph=TemporalGraph, Frames=Frames,Filename=filename, fps=fps)

    # Optionally, load the temporal graph from a file (comment this out if not needed)
    LoadedTemporalGraph, LoadedFrames, fps = LoadTemporalGraph()

    # Optionally Visualize the temporal graph
    # VisualizeTemporalGraph(TemporalGraph=LoadedTemporalGraph, fps=fps)

    # Play the video by traversing the temporal graph
    PlayVideoWithTemporalGraph(TemporalGraph=LoadedTemporalGraph, fps=fps)



if __name__ == "__main__":
    LoadedTemporalGraph, LoadedFrames, fps = LoadTemporalGraph()
    PlayVideoWithTemporalGraph(TemporalGraph=LoadedTemporalGraph, fps=fps)