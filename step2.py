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

    