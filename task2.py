import clip
import torch
import numpy as np
from torchvision.datasets import CIFAR100, CIFAR10
from pkg_resources import packaging
import clip
import os
#deal with images. 
import skimage
#jupyter notebooks. 
import IPython.display
#plotting stuff. 
import matplotlib.pyplot as plt
#image representation for computer vision. 
from PIL import Image
#import cupy as cp
from collections import OrderedDict
import torchvision
def task2():
    """
     Perform linear probing with CLIP on CIFAR-10. 
     You need to extract CLIP image embeddings for 1000 random examples from the training set of CIFAR-10,
     then train a linear classifier. 
     Report classification accuracy on the same 500 random test examples as in Task 1.
    """

    model, preprocess = clip.load("ViT-B/32")
    trainCifar = CIFAR10(os.path.expanduser("~/.cache"),transform = preprocess, train = True, download = True)
    testCifar = CIFAR10(os.path.expanduser("~/.cache"),transform = preprocess, train = False, download = True)
    testExamples = range(0, 500)

    trainExamples = range(0,1000)
    valExamples = range(501, 1001)
    subsetTrain = torch.utils.data.Subset(trainCifar, trainExamples)

    subsetVal = torch.utils.data.Subset(testCifar, valExamples)
    #get a subset of a given dataset. 
    #can't slice into a dataset as you normally would a numpy array. 
    subsetTest = torch.utils.data.Subset(testCifar, testExamples)
    bs = 10
    dataLoaderTrain = torch.utils.data.DataLoader(subsetTrain, batch_size = len(subsetTrain))
    dataLoaderVal = torch.utils.data.DataLoader(subsetVal, batch_size = len(subsetVal))
    dataLoaderTest = torch.utils.data.DataLoader(subsetTest, batch_size =len(subsetTest))
    image_input = next(iter(dataLoaderTrain))[0].cuda()
    val_input = next(iter(dataLoaderVal))[0].cuda()
    test_input = next(iter(dataLoaderTest))[0].cuda()

    labels = next(iter(dataLoaderTrain))[1].cuda()
    val_labels = next(iter(dataLoaderVal))[1].cuda()
    test_labels = next(iter(dataLoaderTest))[1].cuda()

    with torch.no_grad():
        train_image_features = model.encode_image(image_input).float()
        val_image_features = model.encode_image(val_input).float()
        test_image_features = model.encode_image(test_input).float()
    trainDataset = torch.utils.data.TensorDataset(train_image_features, labels)
    valDataset = torch.utils.data.TensorDataset(val_image_features, val_labels)
    testDataset = torch.utils.data.TensorDataset(test_image_features, test_labels)
    trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size = bs, shuffle=True)
    valLoader = torch.utils.data.DataLoader(valDataset,  batch_size = val_image_features.shape[0])
    testLoader = torch.utils.data.DataLoader(testDataset, batch_size = test_image_features.shape[0])
    model = ImageClassifier(train_image_features.shape[-1])
    #can set the betas here. 
    optimizer = torch.optim.Adam(model.parameters(), lr = .001)

    trainLoop(20, model, trainLoader, valLoader, optimizer, catCE)
    sumOutputLoss = 0.0
    sumAcc = 0.0
    count = 0
    for testData in iter(testLoader):
        testImages, testLabels = testData
        outputs = model(testImages)
        loss = catCE(outputs, testLabels)
        acc = calcAccuracy(outputs, testLabels)
        sumOutputLoss+=loss
        sumAcc+=acc
        count+=1
    avgLoss = sumOutputLoss/count
    avgAcc = sumAcc/count
    print("Test loss {} accuracy {}\n".format(avgLoss, avgAcc))
    return 



def trainLoop(numEpochs, model, train_loader, validation_loader, optimizer, loss):
    for epoch in range(numEpochs):
        #set it so that gradient tracking is on
        model.train(True)
        avgLoss = trainEpoch(train_loader,  optimizer, model, loss)

        model.train(False)
        vLossSum = 0
        accSum = 0
        for i, vdata in enumerate(validation_loader):
            vinputs, vlabels = vdata
            voutputs = model(vinputs)
            vLoss = loss(voutputs, vlabels)
            vLossSum+=vLoss
            acc = calcAccuracy(voutputs, vlabels)
            accSum+=acc
        avgVLoss = vLossSum/(i+1)
        avgAcc = accSum/(i+1)
        print("Epoch {}: train {} validation {}\n".format(epoch, avgLoss, avgVLoss))
        print("acc: {}\n".format(avgAcc))
    return 
def trainEpoch(loader,  optimizer, model, lossFxn):
     sumLoss = 0
     lastLoss = 0
     count = 1000
     # iterate through dataloader, 
     # but enumerate means that it keeps track 
     # of which batch we're on. 
     for i, (inputs, labels) in enumerate(loader):
         #inputs,labels = data
         optimizer.zero_grad()
         outputs = model(inputs)

         loss = lossFxn(outputs, labels)
         #computes gradients of the loss function
         loss.backward()
         #takes step of gradient. 
         optimizer.step()
         sumLoss+=loss.item()
         if i%count == count-1:
             lastLoss = sumLoss/count
             sumLoss = 0
     return lastLoss
class ImageClassifier(torch.nn.Module):
    def __init__(self, encodingSize):
        super(ImageClassifier, self).__init__()
        hidden1 = 100
        hidden2 = 50
        self.linear1 = torch.nn.Linear(encodingSize, hidden2).cuda()
        #self.linear2 = torch.nn.Linear(hidden1, hidden2).cuda()
        self.linear3 = torch.nn.Linear(hidden2, 10).cuda()


    def forward(self, x):
        x = torch.nn.functional.relu(self.linear1(x))
        #print("x shape after first linear: ", x.shape)
        #x = torch.nn.functional.relu(self.linear2(x))
        #print("x shape after 2nd linear: ", x.shape)
        x = torch.nn.functional.softmax(self.linear3(x))
        #print("x shape output: ", x.shape)
        return x
def catCE(probabilities, labels):
    """
    Calculates categorical crossentropy on size n x 10 probabilities matrix, 
    and size n true labels array. 
    """
    n = probabilities.shape[0]
    encodedLabels = torch.nn.functional.one_hot(labels, num_classes = 10).cpu().float()
    assert(encodedLabels.shape == (n, 10))
    logProbs = -1* torch.log(probabilities.cpu())

    ceVals = torch.diag(logProbs@encodedLabels.T)
    #print("ce vals: ", ceVals)
    avgCE = torch.mean(ceVals)
    return avgCE
def calcAccuracy(predictions, labels):
    guesses = torch.argmax(predictions.cpu(), axis=-1)
    assert(guesses.shape == (predictions.shape[0],))
    equalArray = labels.cpu() == guesses
    percentEqual = torch.mean(equalArray.float())
    return percentEqual
task2()