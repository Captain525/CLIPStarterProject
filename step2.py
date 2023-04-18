import numpy as np
from matplotlib import image
import torch
import clip
from PIL import Image

from task1 import graphClassifications
import matplotlib.pyplot as plt


import openai
from gpt import GPT, set_openai_key, Example
def pipeline():
    model, preprocess = clip.load("ViT-B/32")
    images = loadImages(100, preprocess)
    objectList, dictTransform = loadObjects()
    api_key = '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'
    openai.api_key = api_key
    
    gpt = GPT(engine = "davinci", temperature = .5, max_tokens = 100)
    feedTrainingExamples(gpt, objectList, 5)
    text_descriptions = [f"This is a photo of a {object}" for object in objectList]

    image_input = torch.tensor(np.stack(images)).cuda()
    text_tokens = clip.tokenize(text_descriptions).cuda()
    with torch.no_grad():
        image_features = model.encode_image(image_input).float()
        text_features = model.encode_text(text_tokens).float()
        text_features/= text_features.norm(dim = -1, keepdim=True)
        image_features /=image_features.norm(dim = -1, keepdim = True)
    #100 is a temperature parameter. 
    text_values = image_features@text_features.T
    #should already be roughly between 0 and 1, bc 1 is a perfect match. 
    print(text_values[0])
    threshold = .9
    print(torch.mean(text_values))
    chosenObjectsThreshold, stringObjectsList = objectsThresholding(text_values, threshold, dictTransform)
    #use a threshold value around .25
    #chosenObjectsThreshold, stringObjectsList = objectThresholdSimple(text_values, threshold, dictTransform)
    print(stringObjectsList)
    numGraphing = 10
   
    captionList = []
    for stringObject in stringObjectsList[0:numGraphing]:
        captionList.append(gpt.get_top_reply(stringObject).replace("output:", "").replace("\n", ""))
    imageInputPermute = image_input[0:numGraphing].permute(0,2,3,1).cpu().numpy()
    chosenObjectsThresholdGraph = chosenObjectsThreshold[0:numGraphing]
    for i in range (0, numGraphing):
        plt.imshow(imageInputPermute[i])
        plt.text(0,0, stringObjectsList[i])
        plt.figtext(0,0,captionList[i])
        plt.show()

    #displayImageWithCategories(chosenObjectsThreshold, image_input[0:6].permute(0,2,3,1).cpu().numpy())


def task3():

    """
    Task 3: Convert an image into a list of objects with CLIP. 
    There are several crucial missing pieces for you to figure out: 
    (1) For the zero-shot classification example in Task 1, each image has only a single label.
    How can you output multiple objects for an image?; 
    (2) What would make a good list of “candidate objects”?
     You might find it helpful to browse some image examples in the Flickr dataset;
    (3) Objects only or also attributes? In your list of candidate objects, do you intend to include “cat”, “dog”, or also “cute cat”, “lazy dog”?
    
    Task 4: Convert a list of objects into a sentence with GPT-3.
    Our notebook already provides some examples for in-context learning, 
    but you would need to decide on the in-context examples to provide to GPT-3.

    Task 5: Run your pipeline on at least 100 images from the validation / test set of Flickr 8k, 
    and summarize your qualitative observations. Are you happy with what you got?
    If not, are there things you would like to try to improve the quality of image captions,
    or do you think there are fundamental limitations on the way we compose CLIP and GPT-3 models?


    """

    #To output multiple objects for an image, could take all classifications over some probability threshold, or you could maybe find some 
    # net similarity between all types of objects, so that we pick those that stnad out from eachother but are most prominent. 
    model, preprocess = clip.load("ViT-B/32")
    images = loadImages(100, preprocess)
    objectList, dictTransform = loadObjects()
    numberClasses = range(0, len(objectList))
    #use these as potential captions to get the images from. Same as in task1.
    text_descriptions = [f"This is a photo of a {object}" for object in objectList]

    image_input = torch.tensor(np.stack(images)).cuda()
    text_tokens = clip.tokenize(text_descriptions).cuda()
    with torch.no_grad():
        image_features = model.encode_image(image_input).float()
        text_features = model.encode_text(text_tokens).float()
        text_features/= text_features.norm(dim = -1, keepdim=True)
        image_features /=image_features.norm(dim = -1, keepdim = True)
    #100 is a temperature parameter. 
    text_values = image_features@text_features.T
    #should already be roughly between 0 and 1, bc 1 is a perfect match. 
    threshold = .85
    print(torch.mean(text_values))
    chosenObjectsThreshold, stringObjectsList = objectsThresholding(text_values, threshold, dictTransform)
    numGraphing = 6
    
    imageInputPermute = image_input[0:numGraphing].permute(0,2,3,1).cpu().numpy()
    chosenObjectsThresholdGraph = chosenObjectsThreshold[0:numGraphing]
    print(len(chosenObjectsThreshold))
    for i in range (0, numGraphing):
        plt.imshow(imageInputPermute[i])
        plt.text(0,0, stringObjectsList[i])
        plt.show()

    #displayImageWithCategories(chosenObjectsThreshold, imageInputPermute)
    return
def task4():
    api_key = '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'
    openai.api_key = api_key
    
    objectList = ["boat, water, person"]

    openai.FineTune.list() 
    prompt = "cat, mouse, cheese, person"
    gpt = GPT(engine = "davinci", temperature = .5, max_tokens = 100)
    set_openai_key(api_key)
    print("top reply here: \n")
    print(gpt.get_top_reply(prompt))
    gpt.add_example((Example('cat', 'The cat looks cute.')))
    gpt.add_example((Example('cat, dog', 'The cat is playing with a black dog')))
    gpt.add_example(Example('cat, dog, mouse', 'The cat is trying to catch a mouse, and the dog watches.'))
    gpt.add_example(Example('mouse, cheese', 'The cute mouse carries a piece of cheese with its arms.'))
    print(gpt.get_top_reply(prompt))
    feedTrainingExamples(gpt, objectList,50)



    print(gpt.get_top_reply(prompt))
    return 



def feedTrainingExamples(gpt, objectList, numExamples):
    with open("Text/Flickr8k.token.txt") as f:
        lines = f.readlines()
        indices = np.random.randint(0, len(lines), numExamples, dtype = int)
        for i in indices:
            line = lines[i]
            objectsInLine = ""
            index = line.find("#")+3
            #print(line[index:])
            for object in objectList:
                if object in line:
                    objectsInLine += object
            gpt.add_example(Example(objectsInLine, line[index:].replace("\n", "")))

def objectsThresholding(text_values, threshold, objectDictionary):
    """
    Uses thresholding mechanic to choose the classes of an object. Takes in the raw text values rather than the softmax ones. 
    However, it does this by normalizing from 0 to 1 based on the min and max of each vector. 
    THRESHOLD = .8 gets pretty good results. 
    """

    maxValues = torch.max(text_values, dim=-1)[0]

    minValues = torch.min(text_values, dim=-1)[0]
    #normalization technique where they're normalized from 0 to 1, but ONLY depend on the max and min, and not the distribution. 
    normalizedValues = (text_values - minValues[:, None])/(maxValues[:, None] - minValues[:, None])

    chooseObject = normalizedValues>=threshold

    nonzero = torch.nonzero(chooseObject)
  
    numberOfEach = torch.bincount(nonzero[:, 0]).cpu()
   
    assert(torch.sum(numberOfEach) == nonzero.shape[0])
    cumSumNum = torch.cumsum(numberOfEach, 0)

    listOfIndividualTensors = torch.tensor_split(nonzero, cumSumNum,dim = 0)
   
    chosenObjects = [list(map(objectDictionary.get, tensor[:, 1].tolist()))for tensor in listOfIndividualTensors]
    stringObjectList=[]
    for objectList in chosenObjects:
        stringObjects = ""
        for object in objectList:
            stringObjects+=object + ", "
        stringObjectList.append(stringObjects[:-2])
    
    return chosenObjects, stringObjectList

def objectThresholdSimple(text_values, threshold, dictTransform):
    """
    Absolute thresholding based on cosine similarity. 
    """
    chooseObject = text_values>=threshold
    nonzero = torch.nonzero(chooseObject)
  
    numberOfEach = torch.bincount(nonzero[:, 0]).cpu()
   
    assert(torch.sum(numberOfEach) == nonzero.shape[0])
    cumSumNum = torch.cumsum(numberOfEach, 0)

    listOfIndividualTensors = torch.tensor_split(nonzero, cumSumNum,dim = 0)
    chosenObjects = [list(map(dictTransform.get, tensor[:, 1].tolist()))for tensor in listOfIndividualTensors]
    stringObjectList=[]
    for objectList in chosenObjects:
        stringObjects = ""
        for object in objectList:
            stringObjects+=object + ", "
        stringObjectList.append(stringObjects[:-2])
    
    return chosenObjects, stringObjectList





def getObjectsForImages(text_probs, objectDictionary):
    """
    From these probabilities, pick which classes are represented.
    Look at ratios? 
    
    Works faster now, the meat of it makes sense, but the actual details of the implementation an dhow i choose the top k objects
    depends. 
    """
    listClassification = []
    listObjectLists = []
    print("text probs shape:", text_probs.shape)
    n = text_probs.shape[0]
    sortedTopLow = torch.argsort(text_probs, axis=-1)
    print(sortedTopLow.dtype)
    inversePerm = torch.zeros(sortedTopLow.shape, dtype = torch.int64)
    print("inverse perm shape: ", inversePerm.shape)
    print("shape is: ", (torch.arange(sortedTopLow.shape[1]).repeat(n,1)).shape)
    #this is the form it was in in the 1d case. 
    print("sorted top low shape: ", sortedTopLow.shape)
    #print("inversePerm sortedTopLow shape: ", inversePerm[:, sortedTopLow].shape)
    inversePerm[torch.arange(n)[:, None], sortedTopLow] = torch.arange(sortedTopLow.shape[1]).repeat(n,1).type(torch.int64)
    print("check that what it's doing is right", inversePerm[torch.arange(n)[:, None], sortedTopLow.cpu()])
    print("inverse perm shape: ", inversePerm.shape)
    #None basically expand dims. 
    sortedProbabilities = text_probs[torch.arange(n)[:, None], sortedTopLow]
    print("sorted prob row: ", sortedProbabilities[0])
    print("sorted probs shape: ", sortedProbabilities.shape)
    rolledProbs = torch.roll(sortedProbabilities, 1, -1)
    print("rolled prob shape: ", rolledProbs.shape)
    ratioTensor = sortedProbabilities/rolledProbs
    print("ratio shape: ", ratioTensor.shape)
    cutoffNum = torch.argmax(ratioTensor[:, 1:], axis=-1)#+ torch.ones(ratioTensor.shape[0])
    print("cutoff shape: ", cutoffNum.shape)
    print("correct shape: ", sortedProbabilities[torch.arange(n), cutoffNum].shape)
    boolClass = (sortedProbabilities >= (sortedProbabilities[torch.arange(n), cutoffNum])[:, None])
    print(boolClass)
    classificationTensor = boolClass[torch.arange(n)[:, None], inversePerm]
    print("class tensor: ", classificationTensor)
    print("class tensor shape: ", classificationTensor.shape)
    nonzero = torch.nonzero(classificationTensor)
    print("nonzero shape: ", nonzero.shape)
    numberOfEach = torch.bincount(nonzero[:, 0]).cpu()
    print("number of each: ", numberOfEach)
    print("number of each shape: ", numberOfEach.shape)
    assert(torch.sum(numberOfEach) == nonzero.shape[0])
    cumSumNum = torch.cumsum(numberOfEach, 0)
    print(cumSumNum)
    print(cumSumNum.shape)
    listOfIndividualTensors = torch.tensor_split(nonzero, cumSumNum,dim = 0)
   
    chosenObjects = [list(map(objectDictionary.get, tensor[:, 1].tolist()))for tensor in listOfIndividualTensors]
    print(chosenObjects)
    """
    Difference approach: 
    #first index will be an-a1 which is negative of the total difference. 
    differenceVector = rolledProbs - sortedProbs
    percentChange = differenceVector/(-differenceVector[0])
    roght now, just do the same cutoffnumber with the argmax. 

    """
    return classificationTensor, chosenObjects

def loadImages(number,preprocess):
    with open("Text/Flickr_8k.testImages.txt") as f:
        lines = f.readlines()
        listImages = [image.strip() for image in lines]
    listImagesarr = np.array(listImages, str)
    print("list images arr: ", listImagesarr)
    sampleImages = listImagesarr[np.random.choice(np.arange(listImagesarr.shape[0]), number, replace = False)]

    print('sample images: ', sampleImages)
    listPaths = ["Flicker8k_Dataset/" + imagePath for imagePath in sampleImages]
    imageList = []
    for path in listPaths:
        imageValue = Image.open(path).convert("RGB")
        imageList.append(preprocess(imageValue))
    return imageList
def displayImageWithCategories(imageObjectLists, image_input):
    """
    Given lists of classification for each image, display the image and the list. 
    """
    plt.figure(figsize=(16, 16))

    for i, image in enumerate(image_input):
        plt.subplot(4, 4, 2 * i + 1)
        plt.imshow(image)
        plt.subplot(4,4,2*i + 2)
        plt.text(0, 0, listMake(imageObjectLists[i]))


    plt.subplots_adjust(wspace=0.5)
    plt.show()
def listMake(listStrings):
    str = ""
    for string in listStrings:
        str+=(string + ", ")
    return str
def getObjects():
    """
    Get the list of objects from the flickr dataset. 
    For now, just do it by going through the text doc and histograms of words, then picking them manually. 
    Maybe use noun detector later. 
    """
    #this one has words without their endings. 
    filePath = "Text/Flickr8k.lemma.token.txt"

    with open(filePath, "r") as f:
        listLines = f.readlines()
        wordList = [[word for word in line.split()[1:]]for line in listLines]
        print(wordList[0:10])
    #have to  invert the order. 
    collapsedList = [word for line in wordList for word in line]
    listArray = np.array(collapsedList, str)
    uniqueElements, counts = np.unique(listArray, return_counts = True)
    indicesToSort = np.argsort(counts)

    sortedElements = uniqueElements[indicesToSort]
    sortedCounts = counts[indicesToSort]
    combinedArray = np.vstack([sortedElements, sortedCounts]).T
    print("size: ", combinedArray.shape)
    print("pairs of words: ", combinedArray[::-1][0:100])
def loadObjects():
    filePath = "Objects.txt"
    with open(filePath, "r") as f:
        listLines = f.readlines()
        listObjects = [word.strip() for word in listLines]
        print(listObjects)
    rangeValues = range(0, len(listObjects))
    assert(len(rangeValues) == len(listObjects))
    dictTransform = dict(zip(rangeValues, listObjects))
    return listObjects, dictTransform
pipeline()