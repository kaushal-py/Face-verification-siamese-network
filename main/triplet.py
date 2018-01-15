import os
import random
import glob
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib import image as mpimg

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

(anchor, positive, negative) = fetch(0)
a = Image.fromarray(anchor, 'RGB')
a.show()
p = Image.fromarray(positive, 'RGB')
p.show()
n = Image.fromarray(negative, 'RGB')
n.show()