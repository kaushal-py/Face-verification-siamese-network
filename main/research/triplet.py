import os
import random
import glob
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import torchvision.datasets as dset
import torchvision.transforms as transforms
from config import Config

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

if True:
    dataset_pre = dset.ImageFolder(root=Config.training_dir_pre)
    dataset_post = dset.ImageFolder(root=Config.training_dir_post)
    number_of_images = len(dataset_pre)
else:
    dataset_pre = dset.ImageFolder(root=Config.testing_dir_pre)
    dataset_post = dset.ImageFolder(root=Config.testing_dir_post)
    number_of_images = len(dataset_pre)

def getItem(idx):
    
    # Anchor
    anchor_tuple = dataset_pre.imgs[idx]
    anchor = Image.open(anchor_tuple[0])
                
    # Positive
    probaility = random.randint(1,100)
    if probaility<101:
        positve_tuple = dataset_post.imgs[idx]
    else:
        positive = dataset_pre.imgs[idx+1]

    positive = Image.open(positve_tuple[0])

    probaility = random.randint(1,100)
    if probaility<60:
        negative_tuple = dataset_pre.imgs[(idx+5)%number_of_images]
    else:
        negative_tuple = dataset_post.imgs[(idx+5)%number_of_images]

    negative = Image.open(negative_tuple[0])
    
    # transform=transforms.Compose([transforms.ToTensor()])

    # anchor = anchor.convert("L")
    # positive = positive.convert("L")
    # negative = negative.convert("L")

    # anchor = transform(anchor)
    # positive = transform(positive)
    # negative = transform(negative)
    
    return anchor, positive, negative

import time
start_time = time.time()
print("started")
for i in range(64):
    getItem(i)
print("--- %s seconds ---" % (time.time() - start_time))
