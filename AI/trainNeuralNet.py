import gc
import json
import os
import time
import pandas as pd
import numpy as np
import tracemalloc
import matplotlib.pyplot as plt
import psutil
from sklearn.model_selection import train_test_split
from multiprocessing import Pool, Manager

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from progressBar import getProgressBar

# Set a random seed for reproducibility
RANDOM_SEED = np.random.randint(0, 1000)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# set torch to use the GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


averageWorkerUtilization = 17 #% average worker utilization

class NeuralNet(nn.Module):
    def __init__(self, layers, dropoutRates, l2Reg, inputActivation, hiddenActivation, outputActivation):
        super(NeuralNet, self).__init__()
        self.layers = nn.ModuleList()
        self.dropoutRates = dropoutRates
        self.l2Reg = l2Reg
        self.patience = 10

        # Add input layer
        self.layers.append(nn.Linear(layers[0], layers[1]))
        self.layers.append(self.getActivation(inputActivation))
        self.layers.append(nn.BatchNorm1d(layers[1]))
        self.layers.append(nn.Dropout(dropoutRates[0]))

        # Add hidden layers
        for i in range(1, len(layers) - 2):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
            self.layers.append(self.getActivation(hiddenActivation))
            self.layers.append(nn.BatchNorm1d(layers[i + 1]))
            dropoutRateIndex = min(i, len(dropoutRates) - 1)
            self.layers.append(nn.Dropout(dropoutRates[dropoutRateIndex]))

        # Add output layer
        self.layers.append(nn.Linear(layers[-2], layers[-1]))
        self.layers.append(self.getActivation(outputActivation))

    def getActivation(self, activation):
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'sigmoid':
            return nn.Sigmoid()
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'softmax':
            return nn.Softmax(dim=1)
        elif activation == 'leakyrelu':
            return nn.LeakyReLU()
        elif activation == 'prelu':
            return nn.PReLU()
        elif activation == 'elu':
            return nn.ELU()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

    def forward(self, x):
        for layer in self.layers:
            if isinstance(layer, nn.BatchNorm1d):
                if x.size(0) == 1:
                    continue  # Skip batch normalization if batch size is 1
            x = layer(x)
        return x

    def fit(self, trainLoader, testLoader, optimizer, lossFn, epochs:int, patience:int, device, index=0):
        try:
            bestAcc = -float("inf")
            self.patience = patience
            patienceCounter = 0
            patienceThreshold = 0.05 / 100
            correctThreshold = 0.05

            for epoch in range(epochs):
                self.train()
                for XBatch, yBatch in trainLoader:
                    XBatch, yBatch = XBatch.to(device, non_blocking=True), yBatch.to(device, non_blocking=True)
                    optimizer.zero_grad()
                    outputs = self(XBatch)
                    loss = lossFn(outputs, yBatch)
                    loss.backward()
                    optimizer.step()

                self.eval()
                totalAccuracy, totalCount = 0.0, 0

                with torch.no_grad():
                    for XBatch, yBatch in testLoader:
                        XBatch, yBatch = XBatch.to(device, non_blocking=True), yBatch.to(device, non_blocking=True)
                        outputs = self(XBatch)
                        # clamp the output to 0-1
                        outputs = torch.clamp(outputs, 0, 1)
                        errors = torch.abs(outputs - yBatch)
                        corrects = (errors < correctThreshold).float()
                        # print(f"Correct: {corrects.sum().item():.4f}")
                        totalAccuracy += corrects.sum().item()
                        totalCount += corrects.numel() 
                        # print(f"Batch correct: {correct.sum().item():.4f}  Total Accuracy So Far: {totalAccuracy:.4f}  Total Count So Far: {totalCount:.0f}")
                        
                accuracy = totalAccuracy / totalCount
                

                # Update best accuracy with patience
                if accuracy - bestAcc > patienceThreshold:
                    bestAcc = accuracy
                    patienceCounter = 0
                else:
                    patienceCounter += 1

                print(f"[{index}] | Epoch {epoch + 1}/{epochs} (Acc: {accuracy*100:.2f}%) (patience: {patience - patienceCounter})..." + " " * 5, end="\r")
                
                if patienceCounter >= patience:
                    break

            return bestAcc
        except KeyboardInterrupt:
            print(f"\nTraining interrupted. Best accuracy: {bestAcc * 100:.2f}%", end="\r")
            return bestAcc


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

def createModel(layers, dropoutRates, l2Reg, learningRate, inputActivation, hiddenActivation, outputActivation, loss, optimizer):
    model = NeuralNet(layers, dropoutRates, l2Reg, inputActivation, hiddenActivation, outputActivation)

    model.to(device)

    if optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learningRate, weight_decay=l2Reg)
    elif optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learningRate, weight_decay=l2Reg)
    elif optimizer == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=learningRate, weight_decay=l2Reg)
    # Add other optimizers as needed

    if loss == 'binary_crossentropy':
        loss = nn.BCELoss()
        
    elif loss == 'mse':
        loss = nn.MSELoss()
        
    elif loss == 'smoothl1':
        loss = nn.SmoothL1Loss()
        
    elif loss == 'l1':
        loss = nn.L1Loss()
        
    elif loss == 'huber':
        loss = nn.HuberLoss()
        
    elif loss == 'crossentropy':
        loss = nn.CrossEntropyLoss()
        
    elif loss == 'nll':
        loss = nn.NLLLoss()
        
    elif loss == 'poisson':
        loss = nn.PoissonNLLLoss()
    # Add other loss functions as needed

    return model, optimizer, loss, device

def evaluateIndividual(params, XTrain, yTrain, XTest, yTest, populationSize, index):
    time.sleep(index * 0.05)  # wait a bit for a better printout

    model, optimizer, lossFn, device = createModel(
        layers=params['layers'],
        dropoutRates=params['dropoutRates'],
        l2Reg=params['l2Reg'],
        learningRate=params['learningRate'],
        inputActivation=params['inputActivation'],
        hiddenActivation=params['hiddenActivation'],
        outputActivation=params['outputActivation'],
        loss=params['loss'],
        optimizer=params['optimizer']
    )

    trainDataset = TensorDataset(torch.tensor(XTrain, dtype=torch.float32), torch.tensor(yTrain, dtype=torch.float32))
    testDataset = TensorDataset(torch.tensor(XTest, dtype=torch.float32), torch.tensor(yTest, dtype=torch.float32))

    trainLoader = DataLoader(trainDataset, batch_size=params['batch_size'], shuffle=True, pin_memory=True)
    testLoader = DataLoader(testDataset, batch_size=params['batch_size'], shuffle=False, pin_memory=True)           

    if params['model'] is not None and not isinstance(params['model'], str):
        model = params['model']
        model = model.to(device)

    bestValAccuracy = model.fit(
        trainLoader,
        testLoader,
        optimizer,
        lossFn,
        params['epochs'],
        params['patience'],
        device,
        index
        )

    params['fitness'] = bestValAccuracy

    model = model.cpu()
    params['model'] = model

    del model
    del trainLoader
    del testLoader
    torch.cuda.empty_cache()
    gc.collect()

    return params

def worker(paramsIndex: tuple):
    params, index, XTrain, yTrain, XTest, yTest, populationSize, cpuPowerLimit, workerQueue, queueLock = paramsIndex

    time.sleep(index * 0.01)  # wait a bit for a better printout

    while True:
        with queueLock:
            currentCpuLoad = psutil.cpu_percent(interval=0.5)
            workersAhead = len(workerQueue)
            estimatedBaseCpuLoad = max(10, currentCpuLoad - averageWorkerUtilization * workersAhead)
            estimatedNewCpuLoad = estimatedBaseCpuLoad + (averageWorkerUtilization * (workersAhead + 1)) - 2 #2% margin of error

            if estimatedNewCpuLoad <= cpuPowerLimit * 100:
                workerQueue.append(index)  # Add to active workers
                break

        print(f"{workerQueue} | Worker {index} waiting for CPU..." + " " * 5, end="\r")
        time.sleep(0.1)  # Wait before rechecking

    try:
        print(f"{workerQueue} |" + " " * 30, end="\r")
        return evaluateIndividual(params, XTrain, yTrain, XTest, yTest, populationSize, index)
    finally:
        with queueLock:
            workerQueue.remove(index)  # Remove from active workers after completion

def evaluatePopulation(population: list[dict], XTrain, yTrain, XTest, yTest, cpuPowerLimit: float, nJobs: int) -> list[dict]:
    print("Evaluating population..." + " " * 5, end="\r")

    manager = Manager()
    workerQueue = manager.list()
    queueLock = manager.Lock()

    populationSize = len(population)

    poolArgs = [
        (params, idx+1, XTrain, yTrain, XTest, yTest, populationSize, cpuPowerLimit, workerQueue, queueLock)
        for idx, params in enumerate(population)
    ]

    if nJobs == 1:
        evaluatedPopulation = [worker(args) for args in poolArgs]
    else:
        with Pool(processes=nJobs, maxtasksperchild=1) as pool:
            evaluatedPopulation = pool.map(worker, poolArgs)

    gc.collect()

    print("Population evaluated. " + " " * 5, end="\r")

    return evaluatedPopulation

def crossover(parent1: dict, parent2: dict):
    child = {}
    for key in parent1.keys():
        if key.lower() in ['model']: # Skip the model
            child[key] = None
        if np.random.rand() < 0.5:
            child[key] = parent1[key]
        else:
            child[key] = parent2[key]
    return child

def mutate(child: dict, mutationRate: float, mutationForce: float):
    """
    Mutate the child parameters with a given mutation rate.
    """

    if np.random.rand() > mutationRate:
        return child

    mutateLow, mutateHigh = 1 - mutationForce, 1 + mutationForce
    
    optimizerChoices = ['adam', 'sgd']
    activationChoices = ['relu', 'sigmoid', 'tanh', 'softmax', 'leakyrelu', 'prelu', 'elu']
    lossChoices = ['binary_crossentropy', 'mse', 'smoothl1', 'l1', 'huber', 'crossentropy', 'nll', 'poisson']

    noMutationKeys = ['fitness', 'model']
    onlyMutationKeys = ['layers']


    def getMutatedValue(value):
        if isinstance(value, int):
            mutation = np.random.normal(mutateLow, mutateHigh)
            return int(abs(mutation * value))
        elif isinstance(value, float):
            mutation = np.random.normal(mutateLow, mutateHigh)
            return round(abs(mutation * value), 6)
        
        # return the original value if not int or float
        return value

    goodKey = False

    while not goodKey:
        randomKey = np.random.choice(list(child.keys()))
        if randomKey.lower() not in noMutationKeys:
            if len(onlyMutationKeys) > 0:
                if randomKey.lower() in onlyMutationKeys:
                    goodKey = True
            else:
                goodKey = True

    if randomKey.lower() == 'layers':
        addLayer = np.random.rand() < mutationRate 
        removeLayer = np.random.rand() < mutationRate / 1.5
        
        hiddenLayerIndex = np.random.randint(1, len(child[randomKey]) - 1)
        minNeurons = 250

        if addLayer:
            print(f"Adding layer at index {hiddenLayerIndex}... " + " "*10, end="\r") # Debug

            if hiddenLayerIndex == 1:
                numHiddenNeurons = int(child[randomKey][1])
            elif hiddenLayerIndex == len(child[randomKey]) - 2:
                numHiddenNeurons = int(child[randomKey][-2])
            else:
                numHiddenNeurons = (int(child[randomKey][hiddenLayerIndex - 1]) + int(child[randomKey][hiddenLayerIndex + 1])) // 2

            child[randomKey].insert(hiddenLayerIndex, np.clip(getMutatedValue(numHiddenNeurons), a_min=minNeurons, a_max=None))

        elif removeLayer and len(child[randomKey]) > 4:
            print(f"Removing layer at index {hiddenLayerIndex}... " + " "*10, end="\r") # Debug
            child[randomKey].pop(hiddenLayerIndex)

        else:
            print(f"Mutating layer at index {hiddenLayerIndex}... " + " "*10, end="\r") # Debug
            child[randomKey][hiddenLayerIndex] = np.clip(getMutatedValue(child[randomKey][hiddenLayerIndex]), a_min=minNeurons, a_max=None)


        
    elif "rate" in randomKey.lower() or "reg" in randomKey.lower():
        print(f"Mutating {randomKey}... " + " "*20, end="\r") # Debug
        if isinstance(child[randomKey], list):
            for i in range(len(child[randomKey])):
                child[randomKey][i] = np.clip(getMutatedValue(child[randomKey][i]), 0.000001, 0.999999)
        else:
            child[randomKey] = np.clip(getMutatedValue(child[randomKey]), 0.000001, 0.999999)

        

    elif randomKey.lower() == 'optimizer':
        print(f"Mutating {randomKey}... " + " "*20, end="\r") # Debug
        choice = child[randomKey]
        while choice == child[randomKey]:
            choice = np.random.choice(optimizerChoices)
        
        child[randomKey] = choice
        

    elif 'activation' in randomKey.lower():
        print(f"Mutating {randomKey}... " + " "*20, end="\r") # Debug
        choice = child[randomKey]
        while choice == child[randomKey]:
            choice = np.random.choice(activationChoices)
        
        child[randomKey] = choice

    # if the loss function is binary_crossentropy, the output activation must be sigmoid
    if child['loss'] == 'binary_crossentropy':
        child['outputActivation'] = 'sigmoid'

    elif randomKey.lower() in ['epochs']:
        print(f"Mutating {randomKey}... " + " "*20, end="\r") # Debug
        child[randomKey] = int(np.clip(getMutatedValue(child[randomKey]), 150, None))


    time.sleep(1) # Debug

    return child

def convertNumpyTypes(obj):
    """
    Convert all numpy types to Python types.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convertNumpyTypes(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convertNumpyTypes(i) for i in obj]
    else:
        return obj


def saveBestModel(modelPath: str, bestParams: dict):
    model = bestParams['model']

    print(f"Saving best model... ", end="\r")
    torch.save(model, modelPath)
    bestParamsPath = modelPath.removesuffix('.pt') + 'Params.json'

    bestParams['model'] = modelPath

    bestParams = {k: convertNumpyTypes(v) for k, v in bestParams.items()}

    with open(bestParamsPath, 'w') as f:
        json.dump(bestParams, f, indent=4)

    print("Best model saved.     ", end="\r")
    return bestParams

def saveHistory(generationHistory: list, historyPath):

    for individual in generationHistory:
        if 'model' in individual:
            individual['model'] = None

    # Convert numpy types to Python types
    generationHistory = [convertNumpyTypes(individual) for individual in generationHistory]
    
    with open(historyPath, 'w') as f:
        json.dump(generationHistory, f, indent=4)


def neuralEvolution(XTrain, yTrain, XTest, yTest, initialParams, generations=50, populationSize=10, mutationRate=0.7, mutationForce=0.2, cpuPowerLimit=0.7, numParallelWorkers=1, modelPath="", historyPath=""):
    """
    Perform Neuro-Evolution to find the best model parameters for the given dataset.

    ---
        
    ### Args:
        - XTrain (`ndarray`): Training data features
        - yTrain (`ndarray`): Training data labels
        - XTest (`ndarray`): Testing data features
        - yTest (`ndarray`): Testing data labels
        - initialParams (`dict`): Initial model parameters
        - generations (`int`): Number of generations
        - populationSize (`int`): Number of individuals in the population
        - mutationRate (`float`): Mutation rate (percentage of the population that will have one of their parameters mutated)
        - mutationForce (`float`): Mutation force (how much to mutate the parameters)
        - cpuPowerLimit (`float`): CPU power limit as a percentage (0.0 - 1.0) (only relevent if not using GPU)
        - numParallelWorkers (`int`): Number of parallel workers (default: 1)
    """
    cpuPowerLimit = min(max(cpuPowerLimit, 0.1), 0.9)
    numParallelWorkers = int(max(1, numParallelWorkers))



    print(f"\nPerforming Neuro-Evolution for {generations} generations with a population size of {populationSize}...\n")

    if device == "cpu":
        print(f"running on CPU, with {numParallelWorkers} parallel{'s' if numParallelWorkers > 1 else ''} worker{'s' if numParallelWorkers > 1 else ''} (CPU Power Limit: ~{cpuPowerLimit * 100:.0f}%)")


    generationHistory = []

    population = [initialParams.copy() for _ in range(populationSize)]

    bestIndividual = population[0]

    progressWheelIndex = 0
    totalTime = 0.0
    completion = 0.0

    print(" " * len(str([0] * numParallelWorkers)) + " |" + " " * 60 +
          f"gen 0/{generations} : " +
          getProgressBar(completion, wheelIndex=progressWheelIndex) + 
          f"Best Accuracy: --.--% / {bestIndividual['fitness'] * 100:.2f}%  |  Time remaining: --h--min --s  -  est. Finish Time: --h-- ", end='\r')

    
    for generation in range(generations):
        startTime = pd.Timestamp.now()

        topPopulation:list[dict] = population[:max(4, populationSize // 10)]

        newPopulation = []
        if populationSize == 1:
            child = mutate(bestIndividual, mutationRate, mutationForce)
            newPopulation.append(child)
        else:
            for _ in range(populationSize):
                print(f"Mutating individual {len(newPopulation) + 1}/{populationSize}... " + " "*25, end="\r")
                parent1, parent2 = np.random.choice(topPopulation, size=2, replace=False)

                child = crossover(parent1, parent2)

                child = mutate(child, mutationRate, mutationForce)

                newPopulation.append(child)


        newPopulation = evaluatePopulation(newPopulation, XTrain, yTrain, XTest, yTest, cpuPowerLimit, numParallelWorkers)

        population = topPopulation + newPopulation
            

        # Combine fitness with a penalty for hidden neurons:
        # Fewer neurons are better when accuracies are similar.
        penaltyFactor = 0.0001  # Adjust this factor as needed
        population.sort(key=lambda x: x['fitness'] - penaltyFactor * sum(x['layers'][1:-1]), reverse=True)

        # Keep the same population size
        population = population[:populationSize]

        tempPopulation = population[0].copy()
        # tempPopulation['model'] = modelPath
        generationHistory.append(tempPopulation)
        saveHistory(generationHistory, historyPath) # TODO bug fix ?

        if population[0]['fitness'] >= bestIndividual['fitness']:
            bestIndividual = saveBestModel(modelPath, bestIndividual)


        # Print progress

        progressWheelIndex += 1
        completion = progressWheelIndex / generations
        endTime = pd.Timestamp.now()
        elapsed = endTime - startTime

        totalTime += elapsed.total_seconds()
        averageTime = totalTime / progressWheelIndex

        estTimeRemaining = averageTime * (generations - progressWheelIndex)
        estHours = int(estTimeRemaining // 3600)
        estMinutes = int((estTimeRemaining % 3600) // 60)
        estSeconds = int(estTimeRemaining % 60)

        estDays = estHours // 24
        estHours = estHours % 24

        if estDays > 0:
            estTimeStr = f"{estDays}day{'s' if estDays > 1 else ''} {estHours:02}h{estMinutes:02}min {estSeconds:02}s"
        else:
            estTimeStr = f"{estHours:02}h{estMinutes:02}min {estSeconds:02}s"

        estFinishTime = pd.Timestamp.now() + pd.Timedelta(seconds=estTimeRemaining)
        if estTimeRemaining > 86400:
            estFinishTimeStr = estFinishTime.strftime('%d %b %Y %H:%M')
        else:
            estFinishTimeStr = f"{estFinishTime.hour:02}h{estFinishTime.minute:02}"

        print(" " * len(str([0] * numParallelWorkers)) + " |" + " " * 60 +
              f"gen {generation + 1}/{generations} : " +
              getProgressBar(completion, wheelIndex=progressWheelIndex) + 
              f"Best Accuracy: {newPopulation[0]['fitness'] * 100:.2f}% / {bestIndividual['fitness'] * 100:.2f}%  |  Time remaining: {estTimeStr}  -  est. Finish Time: {estFinishTimeStr} ", end='\r')

    bestIndividual['model'] = modelPath
    return bestIndividual, generationHistory

def plotResults(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.show()

def main():
    # Check if GPU is available
    print("-" * 50)
    print("System Information:")
    print("PyTorch version:", torch.__version__)
    print(" - Available GPUs:", torch.cuda.device_count())
    print(" - CUDA available:", torch.cuda.is_available())
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
    
    print("-" * 50, "\n")

    # Start memory profiler
    tracemalloc.start()

    rootDir = os.path.dirname(os.path.abspath(__file__))
    
    csvPath = os.path.join(rootDir, 'Bad Apple!!_24x18.csv')
    if not os.path.isfile(csvPath):
        print("Dataset not found. Please generate the dataset using the 'videoToFeatures.py' script.")
        exit(1)
    
    csvName = os.path.basename(csvPath).removesuffix('.csv')
    csvParam = csvName.split('_')[-1]
    width, height = csvParam.split('x')
    try:
        width = int(width)
        height = int(height)
    except ValueError:
        print(f"Invalid video parameter: {csvParam}")
        exit(1)
    
    # Create the base directory relative to the python file root
    baseDir = os.path.join(rootDir, "models", f"{width}x{height}")
    if not os.path.isdir(baseDir):
        print(f"Model directory not found. Creating directory: {baseDir}")
        os.makedirs(baseDir)
    
    bestParamsPath = os.path.join(baseDir, 'bestModelParams.json')
    historyPath = os.path.join(baseDir, 'generationHistory.json')
    modelPath = os.path.join(baseDir, 'bestModel.pt')

    # check if the model directory exists
    if not os.path.isdir(baseDir):
        print(f"Model directory not found. Creating directory: {baseDir}")
        os.makedirs(baseDir)
    
    data = loadData(csvPath)
    resizedFrameData, originalFrameData = preprocessData(data)
    
    
    XTrain, XTest, yTrain, yTest = train_test_split(resizedFrameData, originalFrameData, test_size=0.3, random_state=42)

    inputNeurons = XTrain.shape[1]
    outputNeurons = yTrain.shape[1]

    # create 3 hidden layers with interpolated number of neurons
    hidden1 = (2 * inputNeurons + outputNeurons) // 3
    hidden2 = (inputNeurons + outputNeurons * 2) // 3

    initialParams = {
        'layers': [
            inputNeurons,
            hidden1,
            hidden2,
            outputNeurons
        ],
        'dropoutRates': [
            0.0001
        ],
        'l2Reg': 0.001,
        'learningRate': 0.0006,
        'inputActivation': 'relu',
        'hiddenActivation': 'relu',
        'outputActivation': 'relu',
        'loss': 'huber',
        'optimizer': 'adam',
        'batch_size': 6000,
        'epochs': 150,
        'patience': 10,
        'fitness': 0.0
    }
    
    
    useBestModel = True
    useTrainedModel = True

    # check if there is a saved best model
    if os.path.isfile(bestParamsPath) and useBestModel:
        print("Loading best model parameters...", end=' ')
        with open(bestParamsPath, 'r') as f:
            bestModelParam = json.load(f )
            # check if input neurons and output neurons match the data
            if bestModelParam['layers'][0] == inputNeurons and bestModelParam['layers'][-1] == outputNeurons:
                initialParams = bestModelParam
                print("Done")
                if useTrainedModel:
                    print("Loading trained model...", end=' ')
                    model = torch.load(modelPath, weights_only=False)
                    model = model.to(device)
                    model.eval()
                    initialParams['model'] = model
                    print("Done")
            else:
                print("Invalid model parameters. Using default parameters.")

    

    print("\nLayers configuration:")
    print(f"- Neurons layers: {initialParams['layers']}")
    print(f"- Batch size: {initialParams['batch_size']}")
    print(f"- Epochs: {initialParams['epochs']}")


    generations = 5
    populationSize = 5
    mutationRate = 0 # percentage of the population that will have one of their parameters mutated (does not include the parents of the population)
    mutationForce = 0 # will mutate numerical values by a factor of 1 +/- mutationForce
    cpuPowerLimit = 0.85
    nJobs = 1

    mutationRate = 0 if useTrainedModel else mutationRate
    mutationForce = 0 if useTrainedModel else mutationForce
    populationSize = 1 if useTrainedModel else populationSize
    
    bestParams, generationHistory = neuralEvolution(XTrain=XTrain,
                                                    yTrain=yTrain,
                                                    XTest=XTest,
                                                    yTest=yTest,
                                                    initialParams=initialParams,
                                                    generations=generations,
                                                    populationSize=populationSize,
                                                    mutationRate=mutationRate,
                                                    mutationForce=mutationForce,
                                                    cpuPowerLimit=cpuPowerLimit,
                                                    numParallelWorkers=nJobs,
                                                    modelPath=modelPath,
                                                    historyPath=historyPath
                                                    )    
    
    saveHistory(generationHistory, historyPath)

    print("Neuro-Evolution completed." + " " * 50)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nNeuro-Evolution interrupted.")
        exit(1)