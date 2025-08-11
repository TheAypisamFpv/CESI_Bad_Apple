import os, sys
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
from progressBar import *

# Register the BadAppleModel class as safe for loading
# This is required for PyTorch 2.6+ security features
torch.serialization.add_safe_globals(['customModel.BadAppleModel'])

# Set a random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Set torch to use the GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def preprocessData(data): 
    print("Preprocessing data...", end=" ")
    sys.stdout.flush()
    
    # Convert string representation of lists to numpy arrays
    data['resizedFrameData'] = data['resizedFrameData'].apply(lambda x: np.fromstring(x.strip('[]'), sep=','))
    data['originalFrameData'] = data['originalFrameData'].apply(lambda x: np.fromstring(x.strip('[]'), sep=','))
    
    resizedFrameData = np.stack(data['resizedFrameData'].values)
    originalFrameData = np.stack(data['originalFrameData'].values)
    
    print(f"done. (data length: {len(data)})")
    return resizedFrameData, originalFrameData

def trainModel(model, trainLoader, testLoader, criterion, optimizer, epochs, patience=10, savePath=None, modelHistoryPath=None, startEpoch=0, historyPath=None):
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
        modelHistoryPath: Path to save model checkpoints for each epoch
        startEpoch: Starting epoch number (for continued training)
        historyPath: Path to save/load training history CSV
        
    Returns:
        history: Dictionary containing training history
    """
    history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'train_accuracy': [],
        'val_accuracy': []
    }
    
    # Load existing history if available
    if historyPath and os.path.exists(historyPath):
        try:
            existingHistory = pd.read_csv(historyPath)
            print(f"Loaded existing training history from {historyPath}")
            history['train_loss'] = existingHistory['train_loss'].tolist()
            history['val_loss'] = existingHistory['val_loss'].tolist()
            history['train_accuracy'] = existingHistory['train_accuracy'].tolist()
            history['val_accuracy'] = existingHistory['val_accuracy'].tolist()
            
            # Ensure we only keep history up to startEpoch
            if len(history['train_loss']) > startEpoch:
                for key in history:
                    history[key] = history[key][:startEpoch]
        except Exception as e:
            print(f"Error loading history: {e}")
    
    bestValLoss = min(history['val_loss']) if history['val_loss'] else float('inf')
    bestValAccuracy = max(history['val_accuracy']) if history['val_accuracy'] else 0.0
    bestModel = None
    counter = 0
    correctThreshold = 0.1 # Threshold for considering a prediction correct (Error < 10%)

    trainingStartTime = time.time()
    
    for epoch in range(epochs):
        actualEpoch = startEpoch + epoch + 1
        print(f"\n\nEpoch {actualEpoch}/{startEpoch + epochs} - Training")
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
            
            # Calculate accuracy
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
        print(f"\nEpoch {actualEpoch}/{startEpoch + epochs} - Validation")
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

                print(f"{getProgressBar(completion, wheelIndex=i, maxbarLength=50, style=Style.DOT_GRID)}-> Avg Val Loss and Accuracy: {avgValLoss:.4f} (Best: {bestValLoss:.4f}) | {avgValAccuracy:.4f} (Best: {bestValAccuracy:.4f}) - epoch val ETA: {epochEtaFormatted} - Training time: {trainingElapsedTimeFormatted}    ", end='\r')
    
        # Save history
        history['epoch'].append(actualEpoch)
        history['train_loss'].append(avgTrainLoss)
        history['val_loss'].append(avgValLoss)
        history['train_accuracy'].append(avgTrainAccuracy)
        history['val_accuracy'].append(avgValAccuracy)

        # Save history to CSV after each epoch
        if historyPath:
            historyDf = pd.DataFrame(history)
            historyDf.to_csv(historyPath, index=False)

        if avgValAccuracy > bestValAccuracy:
            bestValAccuracy = avgValAccuracy

        if avgValLoss < bestValLoss:
            bestValLoss = avgValLoss

        if avgValAccuracy == bestValAccuracy or avgValLoss == bestValLoss:
            bestModel = model.state_dict().copy()
            # save the best model
            if savePath is not None:
                bestModelPath = savePath.removesuffix('.pt') + '_best.pt'
                print("\nSaving best model...")
                torch.save(model, bestModelPath)
            
            counter = 0
        else:
            counter += 1
            
        # Save current epoch model
        actualEpoch = startEpoch + epoch + 1
        currentModelPath = os.path.join(modelHistoryPath, f"model_epoch_{actualEpoch}.pt")
        if savePath is not None:
            torch.save(model, currentModelPath)
        
        # Early stopping
        if counter >= patience:
            print(f"\n\nEarly stopping at epoch {actualEpoch}")
            break
        
    print("\nTraining complete")
    return history, bestModel

def saveModelParams(model, inputSize, outputSize, useConvLayers, randomSeed, savePath):
    """Save model parameters to a JSON file."""
    # Convert numpy types to native Python types
    if isinstance(inputSize, tuple):
        inputSize = tuple(int(x) if hasattr(x, 'item') else x for x in inputSize)
    if isinstance(outputSize, tuple):
        outputSize = tuple(int(x) if hasattr(x, 'item') else x for x in outputSize)
    
    params = {
        'inputSize': inputSize,
        'outputSize': outputSize,
        'useConvLayers': bool(useConvLayers),
        'randomSeed': randomSeed
    }
    
    with open(savePath, 'w') as f:
        json.dump(params, f, indent=4)
    
    print(f"Model parameters saved to {savePath}")

def plotTrainingHistory(history, savePath=None):
    """Plot training history and optionally save it to a file."""
    plt.figure(figsize=(12, 5))
    
    # Use epoch numbers for x-axis if available
    epochs = history['epoch'] if 'epoch' in history else range(1, len(history['train_loss']) + 1)
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_accuracy'], label='Train Accuracy')
    plt.plot(epochs, history['val_accuracy'], label='Validation Accuracy')
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
    BATCHSIZE = 96
    EPOCHS = 100
    LEARNINGRATE = 0.002
    TRAINVALSPLIT = 0.8 # 80% training, 20% validation
    PATIENCE = 15
    USECONVLAYERS = True
    
    CONTINUETRAINING = True
    PREVIOUSMODELPATH = r"C:\Users\Aypisam\Documents\Cesi\CESI_Bad_Apple\models\BadAppleModel_48x36_To_480x360_conv\trainingHistory\model_epoch_100.pt"  # Path to a model from a previous epoch to continue training

    CSVDATASETRELATIVEPATH = "dataset\\48x36_To_480x360\\Bad Apple!!_48x36.csv"


    # File paths
    rootDir = os.path.dirname(os.path.abspath(__file__))
    datasetPath = os.path.join(rootDir, CSVDATASETRELATIVEPATH)

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
    print("Loading CSV data...", end=" ")
    data = pd.read_csv(datasetPath)
    print("done.")
    
    inputFrames, outputFrames = preprocessData(data)
    
    # Get input and output dimensions
    inputSize = initialNewFrameWidth, initialNewFrameHeight
    outputSize = outputNewFrameWidth, outputNewFrameHeight

    print(f"Input size: {initialNewFrameWidth}x{initialNewFrameHeight}")
    print(f"Output size: {outputNewFrameWidth}x{outputNewFrameHeight}")

    # Create model directory
    baseModelDir = os.path.join(rootDir, "models", f"BadAppleModel_{initialNewFrameWidth}x{initialNewFrameHeight}_To_{outputNewFrameWidth}x{outputNewFrameHeight}_{'conv' if USECONVLAYERS else 'dense'}")
    
    
    # Create model
    print("Creating model...")
    model = BadAppleModel(inputSize=inputSize, outputSize=outputSize, useConvLayers=USECONVLAYERS)
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNINGRATE)
    print()
    
    # Check if we're continuing training from an existing model
    if CONTINUETRAINING and PREVIOUSMODELPATH and os.path.exists(PREVIOUSMODELPATH):
        print(f"\nLoading previous model from '{PREVIOUSMODELPATH}'...")
        try:
            # Extract epoch number from filename if it follows the pattern model_epoch_X.pt
            if "model_epoch_" in os.path.basename(PREVIOUSMODELPATH):
                epochStr = os.path.basename(PREVIOUSMODELPATH).split("model_epoch_")[1].split(".pt")[0]
                startEpoch = int(epochStr)
                
                modelDir = os.path.dirname(PREVIOUSMODELPATH).removesuffix("\\trainingHistory")
                
                print(f"Starting training from epoch {startEpoch + 1}")
            else:
                modelDir = os.path.dirname(PREVIOUSMODELPATH)

            # Load the model with the appropriate settings for PyTorch 2.6+
            try:
                # First try with weights_only=True (safer, default in PyTorch 2.6+)
                loadedModel = torch.load(PREVIOUSMODELPATH)
                if isinstance(loadedModel, torch.nn.Module):
                    model = loadedModel.to(device)
                else:
                    model.load_state_dict(loadedModel)
                    model = model.to(device)
            except Exception as load_error:
                print(f"First load attempt failed: {load_error}")
                print("Trying alternative loading method...")
                
                # Try with weights_only=False as fallback (less secure, but may work for older models)
                loadedModel = torch.load(PREVIOUSMODELPATH, weights_only=False)
                if isinstance(loadedModel, torch.nn.Module):
                    model = loadedModel.to(device)
                else:
                    model.load_state_dict(loadedModel)
                    model = model.to(device)
                
            # Create a new optimizer with the loaded model parameters
            optimizer = optim.Adam(model.parameters(), lr=LEARNINGRATE)


        except Exception as e:
            print(f"Error loading previous model: {e}")
            print("Starting training from scratch.")
            startEpoch = 0
    else:
        # Check if directory already exists and increment number if needed
        startEpoch = 0
        counter = 0
        modelDir = baseModelDir
        while os.path.exists(modelDir):
            counter += 1
            modelDir = f"{baseModelDir}_{counter}"
    
        os.makedirs(modelDir, exist_ok=True)
        print(f"Created model directory: {modelDir}")
    
    modelHistoryPath = os.path.join(modelDir, "trainingHistory")
    os.makedirs(modelHistoryPath, exist_ok=True)
    

    
    # Print model summary
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Split data into training and validation sets
    XTrain, XTest, yTrain, yTest = train_test_split(
        inputFrames, outputFrames, test_size=1-TRAINVALSPLIT, random_state=RANDOM_SEED
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
        trainDataset, batch_size=BATCHSIZE, shuffle=True, pin_memory=True
    )
    
    testLoader = DataLoader(
        testDataset, batch_size=BATCHSIZE, shuffle=False, pin_memory=True
    )
    
    
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
    historyCSVPath = os.path.join(modelDir, "trainingHistory.csv")
    
    # Train the model
    history, trainedModel = trainModel(
        model=model,
        trainLoader=trainLoader,
        testLoader=testLoader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=EPOCHS,
        patience=PATIENCE,
        savePath=modelPath,
        modelHistoryPath=modelHistoryPath,
        startEpoch=startEpoch,
        historyPath=historyCSVPath
    )
    
    # Save model parameters
    saveModelParams(
        model=trainedModel,
        inputSize=inputSize,
        outputSize=outputSize,
        useConvLayers=USECONVLAYERS,
        randomSeed=RANDOM_SEED,
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
