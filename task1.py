
import numpy as np
import torch
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
from torchvision.datasets import CIFAR100, CIFAR10

def tutorialClip():
    #unsafe but fixes problem with numpy and mkl. 
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    print("torch version", torch.__version__)
    print(clip.available_models())
    print(torch.cuda.is_available())
    #preprocess is a torchvision transform which resizes inputs with center cropping, and normalizes intensity. 
    #model is the model we wish to load. 
    model, preprocess = clip.load("ViT-B/32")
    model.cuda().eval()
    input_resolution = model.visual.input_resolution
    context_length = model.context_length
    vocab_size = model.vocab_size

    print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
    print("Input resolution:", input_resolution)
    print("Context length:", context_length)
    print("Vocab size:", vocab_size)

    tokens =  clip.tokenize("hello world!")
    print(tokens)
   
    # images in skimage to use and their textual descriptions
    descriptions = {
        "page": "a page of text about segmentation",
        "chelsea": "a facial photo of a tabby cat",
        "astronaut": "a portrait of an astronaut with the American flag",
        "rocket": "a rocket standing on a launchpad",
        "motorcycle_right": "a red motorcycle standing in a garage",
        "camera": "a person looking at a camera on a tripod",
        "horse": "a black-and-white silhouette of a horse", 
        "coffee": "a cup of coffee on a saucer"
    }
    original_images = []
    images = []
    texts = []
    plt.figure(figsize=(16, 5))

    for filename in [filename for filename in os.listdir(skimage.data_dir) if filename.endswith(".png") or filename.endswith(".jpg")]:
        name = os.path.splitext(filename)[0]
        if name not in descriptions:
            continue

        image = Image.open(os.path.join(skimage.data_dir, filename)).convert("RGB")
  
        plt.subplot(2, 4, len(images) + 1)
        plt.imshow(image)
        plt.title(f"{filename}\n{descriptions[name]}")
        plt.xticks([])
        plt.yticks([])

        original_images.append(image)
        images.append(preprocess(image))
        texts.append(descriptions[name])

    plt.tight_layout()
    plt.show()

    #build features

    #cuda saves the tensor on the gpu
    image_input = torch.tensor(np.stack(images)).cuda()
    #tokenize into right form for the clip model. 
    #
    text_tokens = clip.tokenize(["This is " + desc for desc in texts]).cuda()
    #requires_grad - variable when making tensors, stating whether we need the gradient. 
    #disables gradient calculation, makes all computations not calculate gradient. 
    #EVEN IF HT EINPUT HAS REQUIRES GRAD TRUE, this call makes it false. 
    #use vision transformer/ text transformer. 
    with torch.no_grad():
        image_features = model.encode_image(image_input).float()
        text_features = model.encode_text(text_tokens).float()
    #calc cos similarity: 
    image_features/=image_features.norm(dim=-1, keepdim = True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    print("image features shape: ", image_features.shape)
    print("text features shape: ", text_features.shape)
    #calculating the similarity between the image features and the text features. 
    similarity = text_features.cpu().numpy()@(image_features.cpu().numpy().T)
    #make the plot of the cosing similarity. 
    count = len(descriptions)
    plt.figure(figsize=(20, 14))
    plt.imshow(similarity, vmin=0.1, vmax=0.3)
    # plt.colorbar()
    plt.yticks(range(count), texts, fontsize=18)
    plt.xticks([])
    for i, image in enumerate(original_images):
        plt.imshow(image, extent=(i - 0.5, i + 0.5, -1.6, -0.6), origin="lower")
    for x in range(similarity.shape[1]):
        for y in range(similarity.shape[0]):
            plt.text(x, y, f"{similarity[y, x]:.2f}", ha="center", va="center", size=12)

    for side in ["left", "top", "right", "bottom"]:
        plt.gca().spines[side].set_visible(False)

    plt.xlim([-0.5, count - 0.5])
    plt.ylim([count + 0.5, -2])
    print("near end")
    plt.title("Cosine similarity between text and image features", size=20)
    plt.show()
    #zero shot classification with cifar 100 dataset. 
    #applies the transform from before. 
    cifar100 = CIFAR100(os.path.expanduser("~/.cache"), transform=preprocess, download=True)
    text_descriptions = [f"This is a photo of a {label}" for label in cifar100.classes]
    #use the names of labels as tokens. 
    text_tokens = clip.tokenize(text_descriptions).cuda()
    #enocde text with clip model. 
    with torch.no_grad():
        text_features = model.encode_text(text_tokens).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)
    #calculates cosine similarity with new text features and previous image features. 
    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    #picks top k probabilities/labels. 
    top_probs, top_labels = text_probs.cpu().topk(5, dim=-1)
    #graph results. 
    graphClassifications(cifar100.classes, original_images, top_probs, top_labels)
   
def task1():
    """
    Task 1: 
    Perform zero-shot image classification with CLIP on CIFAR-10. 
    You only need to pick 500 random examples from the validation / test set of 
    CIFAR-10.

    Idea: Convert the labels to captions, use the model to convert images and 10 class "captions" into vectors, 
    then find cosine similarity between the vectors. After this, we take softmax to pick the correct label. 

    NOTE: NEed to check if the crossentropy loss values are correct. Got 1.65, seems a bit high. 
    """
    #load specific clip model with the corresponding preprocess step. 
    model, preprocess = clip.load("ViT-B/32")
    #already executing the preprocess step and converting from PIL image to tensors. 
    #why does it seem like the preprocess method puts some values OVER 0 or 1. 
    #THEY LOOK WEIRD BC THE PREPROCESSING IS DONE FOR MATH BUT THESE MESS EVERYTHING UP. THE CALCULATIONS ARE FINE. 
    cifar = CIFAR10(os.path.expanduser("~/.cache"),transform = preprocess, train = False, download = True)

    numExamples = range(0, 500)
    #get a subset of a given dataset. 
    #can't slice into a dataset as you normally would a numpy array. 
    subsetTest = torch.utils.data.Subset(cifar, numExamples)
    dataLoader = torch.utils.data.DataLoader(subsetTest, batch_size =len(subsetTest))
    #gets the entire dataset of size 500, with each being an image. Want to preprocess this as well. NOt sure if this is right. 
    #imageTensor = next(iter(dataLoader))[0].cuda()
    
    print(subsetTest[1][1])
    print("max value in tensor: ", torch.max(subsetTest[1][0]))
    plt.imshow(subsetTest[2][0].permute(1,2,0).numpy())
    plt.show()
    
    image_input = next(iter(dataLoader))[0].cuda()
    labels = next(iter(dataLoader))[1].cuda()
    print(image_input.shape)  
    print(labels.shape)
  
    #want to make captions with the cifar labels. 
    text_descriptions = [f"This is a photo of a {label}" for label in cifar.classes]
    
    tokenized_text = clip.tokenize(text_descriptions).cuda()
    with torch.no_grad():
        image_features = model.encode_image(image_input).float()
        text_features = model.encode_text(tokenized_text).float()
        print("shape of image features: ", image_features.shape)
        print("shape of text features: ", text_features.shape)
        #normalize these vectors. 
        text_features/= text_features.norm(dim = -1, keepdim=True)
        image_features /=image_features.norm(dim = -1, keepdim = True)
    #cosine similarity * 100 thru softmax. 
    #softmax over each example to get probabilities. 
    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    assert(text_probs.shape == (500, 10))
    top_probs, top_labels = text_probs.cpu().topk(5, dim=-1)
    numGraphing = 6
    graphClassifications(cifar, image_input[0:numGraphing].permute(0, 2,3,1).cpu().numpy(), top_probs[0:numGraphing], top_labels[0:numGraphing])


    #calculate the loss as an L2 calculation with one hot values. 
    encodedLabels = torch.nn.functional.one_hot(labels, num_classes = 10).cpu().float()
    print("encoded label size: ", encodedLabels.shape)
    print("text probs size: ", text_probs.cpu().shape)
    loss = torch.nn.CrossEntropyLoss(reduction = "mean")
    print(text_probs.cpu()[0:10].numpy())
    print(encodedLabels.cpu()[0:10])
    #they use softmax again I think. 
    lossValue = loss(text_probs.cpu(), labels.cpu())
    ceLoss = catCE(text_probs, labels)
    print("loss value was: ", lossValue)
    print("ce loss was: ", ceLoss)
    accuracy = calcAccuracy(text_probs, labels)
    print("accuracy is: ", accuracy)
    return

def catCE(probabilities, labels):
    """
    Calculates categorical crossentropy on size n x 10 probabilities matrix, 
    and size n true labels array. 
    """
    n = probabilities.shape[0]
    encodedLabels = torch.nn.functional.one_hot(labels, num_classes = 10).cpu().float()
    assert(encodedLabels.shape == (n, 10))
    logProbs = -1* torch.log(probabilities.cpu())

    ceVals = np.diag(logProbs@encodedLabels.T)
    print("ce vals: ", ceVals)
    avgCE = np.mean(ceVals)
    return avgCE
def calcAccuracy(predictions, labels):
    guesses = torch.argmax(predictions.cpu(), axis=-1)
    assert(guesses.shape == (predictions.shape[0],))
    equalArray = labels.cpu() == guesses
    percentEqual = torch.mean(equalArray.float())
    return percentEqual
def graphClassifications(labels, original_images, top_probs, top_labels):
    plt.figure(figsize=(16, 16))

    for i, image in enumerate(original_images):
        plt.subplot(4, 4, 2 * i + 1)
        plt.imshow(image)
        plt.axis("off")

        plt.subplot(4, 4, 2 * i + 2)
        y = np.arange(top_probs.shape[-1])
        plt.grid()
        plt.barh(y, top_probs[i])
        plt.gca().invert_yaxis()
        plt.gca().set_axisbelow(True)
        plt.yticks(y, [labels[index] for index in top_labels[i].numpy()])
        plt.xlabel("probability")

    plt.subplots_adjust(wspace=0.5)
    plt.show()
#task1()