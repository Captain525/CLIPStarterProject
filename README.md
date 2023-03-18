


Notes on step 2: 
We want to get a list of objects which the images in the dataset could be, then pick potentially multiple classes for an object: in essence, a multilabel classification problem. 

First idea was to take the probability distributions and look at how the probabilities change. The intuition for this is that if there are multiple objects in an image, they will likely ALL have high probabilities, and thus the place where there's the biggest "Jump" in probability is probably a good threshold to go from in the classification to not. The reason why I picked this was I wanted to avoid the scenario of redundant classifications, but this didn't really help with that anyway. Also, the problem with this was that it just wasn't that intuitive, and the softmax function called on the results meant that individual values would be effected by the magnitude of others, so if there's a dog and a car but the dog is more prominent, the car will be hurt because of it. 