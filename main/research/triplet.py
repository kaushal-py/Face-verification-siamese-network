import os
import random
import glob
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import torchvision.datasets as dset

def fetch(i):
    cwd = os.getcwd()
    pre = cwd+"/.eycdata/train/pre"
    post = cwd+"/.eycdata/train/post"
    curr = pre
    # Anchor
    person = sorted(os.listdir(curr))[i]
    fol = curr+"/"+person
    for file in os.listdir(fol):
        anchor = file
        anchor = mpimg.imread(os.path.join(fol,anchor))
        
    # Positive
    probaility = random.randint(1,100)
    if probaility>80:
        if curr == pre:
            curr = post
        else:
            curr = pre
    else:
        if curr == pre:
            curr = pre
        else:
            curr = post
    pos_fol = curr+"/augmented/"+person
    positive = random.choice(os.listdir(pos_fol))
    positive = mpimg.imread(os.path.join(pos_fol,positive))
    
    # Negative
    probaility = random.randint(1,10)
    if probaility>6:
        if curr == pre:
            curr = post
        else:
            curr = pre
    else:
        if curr == pre:
            curr = pre
        else:
            curr = post
    fol = curr+"/augmented/"
    while True:
        neg_fol = random.choice(os.listdir(fol))
        if neg_fol != pos_fol:
            break
    negative = random.choice(os.listdir(fol+neg_fol))
    negative = mpimg.imread(os.path.join(fol+neg_fol,negative))
    
    return anchor, positive, negative

# (anchor, positive, negative) = fetch(0)
# a = Image.fromarray(anchor, 'RGB')
# a.show()
# p = Image.fromarray(positive, 'RGB')
# p.show()
# n = Image.fromarray(negative, 'RGB')
# n.show()

def getItem(idx):
    # get pre and post data folders
    pre_data = dset.ImageFolder(root=".eycdata/train/pre/augmented")
    post_data = dset.ImageFolder(root=".eycdata/train/post/augmented")

    # total number of images (classes * image_per_class)
    noOfimages = len(pre_data)

    # get the anchor image from the index
    img_0 = pre_data.imgs[idx]
    # print(img_0, noOfimages)

    # If the second image is same or not
    isSame = random.randint(0,1)
    label = isSame # label = 1 (for same image)

    if isSame:
        # decide if second image should be pre or post
        isPre = random.randint(0,1)
        if isPre:
            img_1 = pre_data.imgs[idx]
        else:
            img_1 = post_data.imgs[idx]
    
    else:
        # decide if second image should be pre or post
        isPre = random.randint(0,1)
        if isPre:
            img_1 = pre_data.imgs[(idx+5)%noOfimages]
        else:
            img_1 = post_data.imgs[(idx+5)%noOfimages] 

    img_0 = Image.open(img_0[0])           
    img_1 = Image.open(img_1[0])           

    print(img_0, img_1, label)

getItem(4)
