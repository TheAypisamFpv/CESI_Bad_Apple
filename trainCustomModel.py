import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import json
from customModel import BadAppleModel
from progressBar import getProgressBar, Style

# Set a random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Set torch to use the GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def loadData(filepath):
    print("Loading CSV data...", end=" ")
    data = pd.read_csv(filepath)
    print("done.")
    return data

def preprocessData(data):
    print("Preprocessing data...", end=" ")
    
    # Convert string representation of lists to numpy arrays
    data['resizedFrameData'] = data['resizedFrameData'].apply(lambda x: np.fromstring(x.strip('[]'), sep=','))
    data['originalFrameData'] = data['originalFrameData'].apply(lambda x: np.fromstring(x.strip('[]'), sep=','))
    
    resizedFrameData = np.stack(data['resizedFrameData'].values)
    originalFrameData = np.stack(data['originalFrameData'].values)
    
    print(f"done. (data length: {len(data)})")
    return resizedFrameData, originalFrameData

def trainModel(model, trainLoader, testLoader, criterion, optimizer, epochs, patience=10, savePath=None, modelHistoryPath=None):
    """
    Train the model with early stopping based on validation loss.
    
    Args:
        model: The neural network model to train
        trainLoader: DataLoader for training data
        testLoader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        epochs: Maximum number of epochs to train
        patience: Number of epochs to wait before early stopping
        savePath: Path to save the best model
        
    Returns:
        history: Dictionary containing training history
    """
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_accuracy': [],
        'val_accuracy': []
    }
    
    bestValLoss = float('inf')
    bestModel = None
    counter = 0
    correctThreshold = 0.05  # Same threshold used in trainNeuralNet.py
    
    trainingStartTime = time.time()
    
    for epoch in range(epochs):
        print(f"\n\nEpoch {epoch + 1}/{epochs} - Training")
        epochStartTime = time.time()
        
        # Training phase
        model.train()
        testLoss = 0.0
        trainCorrect = 0
        trainTotal = 0

        for i, (XBatch, yBatch) in enumerate(trainLoader):
            completion = (i + 1) / len(trainLoader)
            # Move tensors to the configured device
            XBatch, yBatch = XBatch.to(device, non_blocking=True), yBatch.to(device, non_blocking=True)
            
            # Forward pass
            outputs = model(XBatch)
            loss = criterion(outputs, yBatch)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            testLoss += loss.item() * XBatch.size(0)
            avgTrainLoss = testLoss / (i + 1)
            
            # Calculate accuracy (using threshold as in trainNeuralNet.py)
            with torch.no_grad():
                outputs = torch.clamp(outputs, 0, 1)
                errors = torch.abs(outputs - yBatch)
                corrects = (errors < correctThreshold).float()
                trainCorrect += corrects.sum().item()
                trainTotal += corrects.numel()
                
            # Calculate average training loss and accuracy
            trainAccuracy = trainCorrect / trainTotal if trainTotal > 0 else 0
            avgTrainAccuracy = trainAccuracy / (i + 1) 
            
            
            # ETA calculation
            epochElapsedTime = time.time() - epochStartTime
            batchesDone = i + 1
            batchesLeft = len(trainLoader) - batchesDone
            timePerBatch = epochElapsedTime / batchesDone
            eta = batchesLeft * timePerBatch
            completionTime = time.time() + eta
            epochEtaFormatted = time.strftime('%a %d %H:%M:%S', time.localtime(completionTime))

            trainingElapsedTime = time.time() - trainingStartTime
            hours, remainder = divmod(int(trainingElapsedTime), 3600)
            minutes, seconds = divmod(remainder, 60)
            trainingElapsedTimeFormatted = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

            print(f"{getProgressBar(completion, wheelIndex=i, maxbarLength=50, style=Style.DOT_GRID)}-> Avg Train Loss and Accuracy: {avgTrainLoss:.4f} | {avgTrainAccuracy:.4f} - epoch train ETA: {epochEtaFormatted} - Training time: {trainingElapsedTimeFormatted}    ", end='\r')

        # Validation phase
        print(f"\nEpoch {epoch + 1}/{epochs} - Validation")
        model.eval()
        valLoss = 0.0
        valCorrect = 0
        valTotal = 0
        
        with torch.no_grad():
            for i, (XBatch, yBatch) in enumerate(testLoader):
                completion = (i + 1) / len(testLoader)
                
                XBatch, yBatch = XBatch.to(device, non_blocking=True), yBatch.to(device, non_blocking=True)
                
                outputs = model(XBatch)
                loss = criterion(outputs, yBatch)
                
                valLoss += loss.item() * XBatch.size(0)
                avgValLoss = valLoss / (i + 1)
                
                # Calculate accuracy
                outputs = torch.clamp(outputs, 0, 1)
                errors = torch.abs(outputs - yBatch)
                corrects = (errors < correctThreshold).float()
                valCorrect += corrects.sum().item()
                valTotal += corrects.numel()
                
                # Calculate average validation loss and accuracy
                valAccuracy = valCorrect / valTotal if valTotal > 0 else 0      
                avgValAccuracy = valAccuracy / (i + 1)          
                
                # ETA calculation
                epochElapsedTime = time.time() - epochStartTime
                batchesDone = i + 1
                batchesLeft = len(testLoader) - batchesDone
                timePerBatch = epochElapsedTime / batchesDone
                eta = batchesLeft * timePerBatch
                completionTime = time.time() + eta
                epochEtaFormatted = time.strftime('%a %d %H:%M:%S', time.localtime(completionTime))

                trainingElapsedTime = time.time() - trainingStartTime
                hours, remainder = divmod(int(trainingElapsedTime), 3600)
                minutes, seconds = divmod(remainder, 60)
                trainingElapsedTimeFormatted = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

                print(f"{getProgressBar(completion, wheelIndex=i, maxbarLength=50, style=Style.DOT_GRID)}-> Avg Val Loss and Accuracy: {avgValLoss:.4f} (Best: {bestValLoss:.4f}) | {avgValAccuracy:.4f} - epoch val ETA: {epochEtaFormatted} - Training time: {trainingElapsedTimeFormatted}    ", end='\r')
    
        # Save history
        history['train_loss'].append(avgTrainLoss)
        history['val_loss'].append(avgValLoss)
        history['train_accuracy'].append(avgTrainAccuracy)
        history['val_accuracy'].append(avgValAccuracy)

        # Early stopping
        if avgValLoss < bestValLoss:
            bestValLoss = avgValLoss
            bestModel = model.state_dict().copy()
            # save the best model
            if savePath is not None:
                bestModelPath = savePath.removesuffix('.pt') + '_best.pt'
                print("\nSaving best model...")
                torch.save(model, bestModelPath)
            
            counter = 0
        else:
            counter += 1
            
        currentModelPath = os.path.join(modelHistoryPath, f"model_epoch_{epoch + 1}.pt")
        if savePath is not None:
            torch.save(model, currentModelPath)
        
        
        if counter >= patience:
            print(f"\n\nEarly stopping at epoch {epoch+1}")
            break
    
    # Load the best model
    if bestModel is not None:
        model.load_state_dict(bestModel)
        
    # Save the best model if a path is provided
    if savePath is not None:
        torch.save(model, savePath)
        print(f"\nModel saved to {savePath}")
        
    print("\nTraining complete")
    return history, model

def saveModelParams(model, inputSize, outputSize, useConvLayers, savePath):
    """Save model parameters to a JSON file."""
    params = {
        'inputSize': inputSize,
        'outputSize': outputSize,
        'useConvLayers': useConvLayers,
        'architecture': str(model)
    }
    
    with open(savePath, 'w') as f:
        json.dump(params, f, indent=4)
    
    print(f"Model parameters saved to {savePath}")

def plotTrainingHistory(history, savePath=None):
    """Plot training history and optionally save it to a file."""
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    
    if savePath:
        plt.savefig(savePath)
        print(f"Training history plot saved to {savePath}")
    
    plt.show()

def main():
    # Parameters
    batchSize = 64
    epochs = 10
    learningRate = 0.002
    trainValSplit = 0.8 # 80% training, 20% validation
    patience = 15
    useConvLayers = True  # Set to True to use convolutional approach

    CsvDatasetRelativePath = "dataset/24x18_To_480x360/Bad Apple!!_24x18.csv"

    # File paths
    rootDir = os.path.dirname(os.path.abspath(__file__))
    datasetPath = os.path.join(rootDir, CsvDatasetRelativePath)

    # Check if dataset exists
    if not os.path.isfile(datasetPath):
        print(f"Dataset not found at {datasetPath}")
        print("Please run 'videoToFeatures.py' first to generate the dataset.")
        return 1
    
    # Check if size CSV exists
    csvName = os.path.basename(datasetPath).removesuffix('.csv')
    sizeCsvPath = os.path.join(os.path.dirname(datasetPath), f"{csvName.split('_')[0]}_size.csv")
    if not os.path.isfile(sizeCsvPath):
        print(f"Size CSV not found at {sizeCsvPath}. Please ensure the dataset is generated correctly.")
        return 2
    
    # Read size data from CSV
    sizeData = pd.read_csv(sizeCsvPath)
    initialNewFrameWidth = sizeData['initialNewFrameWidth'].values[0]
    initialNewFrameHeight = sizeData['initialNewFrameHeight'].values[0]
    outputNewFrameWidth = sizeData['outputNewFrameWidth'].values[0]
    outputNewFrameHeight = sizeData['outputNewFrameHeight'].values[0]

    # Load and preprocess data
    data = loadData(datasetPath)
    inputFrames, outputFrames = preprocessData(data)
    
    # Get input and output dimensions
    inputSize = initialNewFrameWidth, initialNewFrameHeight
    outputSize = outputNewFrameWidth, outputNewFrameHeight

    print(f"Input size: {initialNewFrameWidth}x{initialNewFrameHeight}")
    print(f"Output size: {outputNewFrameWidth}x{outputNewFrameHeight}")

    # Create model directory
    modelDir = os.path.join(rootDir, "models", f"BadAppleModel_{initialNewFrameWidth}x{initialNewFrameHeight}_To_{outputNewFrameWidth}x{outputNewFrameHeight}")
    os.makedirs(modelDir, exist_ok=True)
    
    modelHistoryPath = os.path.join(modelDir, "trainingHistory")
    os.makedirs(modelHistoryPath, exist_ok=True)
    
    # Create model
    model = BadAppleModel(inputSize=inputSize, outputSize=outputSize, useConvLayers=useConvLayers)
    model = model.to(device)
    
    # Print model summary
    print("\nModel Architecture:")
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Split data into training and validation sets
    XTrain, XTest, yTrain, yTest = train_test_split(
        inputFrames, outputFrames, test_size=1-trainValSplit, random_state=RANDOM_SEED
    )
    
    # Create data loaders
    trainDataset = TensorDataset(
        torch.tensor(XTrain, dtype=torch.float32),
        torch.tensor(yTrain, dtype=torch.float32)
    )
    
    testDataset = TensorDataset(
        torch.tensor(XTest, dtype=torch.float32),
        torch.tensor(yTest, dtype=torch.float32)
    )
    
    trainLoader = DataLoader(
        trainDataset, batch_size=batchSize, shuffle=True, pin_memory=True
    )
    
    testLoader = DataLoader(
        testDataset, batch_size=batchSize, shuffle=False, pin_memory=True
    )
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learningRate)
    
    # Check if GPU is available
    print("-" * 50)
    print("System Information:")
    print("PyTorch version:", torch.__version__)
    print(" - CUDA available:", torch.cuda.is_available())
    
    if torch.cuda.is_available():
        print(" - Available GPUs:", torch.cuda.device_count())
        print(" - Current CUDA device id:", torch.cuda.current_device())
        deviceProps = torch.cuda.get_device_properties(torch.cuda.current_device())
        print(" - CUDA device name:", deviceProps.name)
        
        # Calculate total CUDA cores based on SM count and cores per SM.
        smCount = deviceProps.multi_processor_count
        # For many NVIDIA GPUs (e.g., Turing architecture used in RTX 2060), each SM has 64 cores.
        # You might need to adjust cores_per_sm based on your GPU architecture.
        coresPerSm = 64 if ("RTX" in deviceProps.name or "Turing" in deviceProps.name) else 0
        totalCudaCores = smCount * coresPerSm if coresPerSm else "Unknown"
        print(" - CUDA cores count:", totalCudaCores)

        # Add tensor core count calculation and print
        if isinstance(totalCudaCores, int):
            # Assumption: for Turing architecture or similar, each SM has 8 Tensor Cores.
            tensorCoreCount = smCount * 8
        else:
            tensorCoreCount = "Unknown"
        print(" - Tensor Core count:", tensorCoreCount)
    else:
        print(" - Running on CPU")
    
    print("-" * 50)
    print("\nStarting training...")
    
    # Set file paths for saving
    modelPath = os.path.join(modelDir, "badAppleModel.pt")
    paramsPath = os.path.join(modelDir, "modelParams.json")
    historyPlotPath = os.path.join(modelDir, "trainingHistory.png")
    
    # Train the model
    history, trainedModel = trainModel(
        model=model,
        trainLoader=trainLoader,
        testLoader=testLoader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=epochs,
        patience=patience,
        savePath=modelPath,
        modelHistoryPath=modelHistoryPath
    )
    
    # Save model parameters
    saveModelParams(
        model=trainedModel,
        inputSize=inputSize,
        outputSize=outputSize,
        useConvLayers=useConvLayers,
        savePath=paramsPath
    )
    
    # Plot and save training history
    plotTrainingHistory(history, savePath=historyPlotPath)
    
    print(f"Model training completed. Model saved to {modelPath}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        exit(0)
