import numpy as np
from matplotlib import image
import clip
from PIL import Image
import torch
from task1 import graphClassifications


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
    
    Task 6: Run your pipeline on at least 500 images from the validation / test set of Flickr 8k, 
    and compute at least one quantitative metric of your choice.
    How does your result compare with the previously published results on Flickr 8k image captioning? 
    Do you find tweaking the list of objects (for CLIP) or the in-context examples (for GPT-3) make any quantitative differences?

    
    Now that we have built a quantitative evaluation pipeline, we can try to improve the models and see if they lead to quantitative improvements. 
    This step is completely open-ended, but a few options are available for your consideration: (1) Leverage CLIP image embeddings,
    for example by training an RNN / Transformer decoder (recall your deep learning homework) for image captioning. 
    You can use the training split of Flickr 8k for this purpose; 
    (2) Turn the image captioning framework from Step 2 into a visual question answering model. 
    You should be able to achieve this by tweaking the in-context examples for GPT-3. 
    You are encouraged to perform quantitative evaluations, but qualitative evaluations are also okay.
    Task 7: Pick one of the options above or use your imagination to explore something new (e.g. replace GPT-3 with ChatGPT).


    """

    #To output multiple objects for an image, could take all classifications over some probability threshold, or you could maybe find some 
    # net similarity between all types of objects, so that we pick those that stnad out from eachother but are most prominent. 
    model, preprocess = clip.load("ViT-B/32")
    images = loadImages(100, preprocess)
    objectList = loadObjects()
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
    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    k = 5
    top_probs, top_labels = text_probs.cpu().topk(k, dim=-1)
    print(text_probs)
    numGraphing = 6
    graphClassifications(objectList, image_input[0:numGraphing].permute(0,2,3,1).cpu().numpy(), top_probs, top_labels)


def getObjectsForImages(text_probs):
    """
    From these probabilities, pick which classes are represented.
    Look at ratios? 

    SLOW but whatever, workign on it right now. 
    """
    listClassification = []
    for imageIndex in range(text_probs.shape[0]):
        rowProb = text_probs[imageIndex, :]
        sortedTopLow = np.argsort(rowProb)[::-1]
        inversePerm = np.zeros(shape =(sortedTopLow.shape[0],))
        inversePerm[sortedTopLow] = np.arange(sortedTopLow.shape[0],)
        sortedProbs = rowProb[sortedTopLow]

        """
        RATIO APPROACH: 
        """
        rolledProbs = np.roll(sortedProbs, 1)
        assert(rolledProbs[0] == sortedProbs[-1])
        ratioVector = sortedProbs/rolledProbs
        cutoffNumber = np.argmax(ratioVector[1:]) +1
        boolClass = (sortedProbs>=sortedProbs[cutoffNumber-1]).astype(int)
        classificationVector = boolClass[inversePerm]
        listClassification.append(classificationVector)
        """
        Difference approach: 
        #first index will be an-a1 which is negative of the total difference. 
        differenceVector = rolledProbs - sortedProbs
        percentChange = differenceVector/(-differenceVector[0])
        roght now, just do the same cutoffnumber with the argmax. 

        """
    return np.stack(listClassification, int)

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

    return listObjects
task3()