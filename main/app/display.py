from flask import Flask, render_template, request
import os
from torch.autograd import Variable
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import torch
from siameseContrastive import SiameseNetwork
from appDataset import AppDataset
from torch.utils.data import DataLoader,Dataset
import numpy as np
import zipfile
import os

app = Flask(__name__)

pre_path = "static/upload/pre/temp"
post_path = "static/upload/post/temp"

# Load model
net = torch.load('model.pt').eval()

@app.route("/")
def display():
    return render_template('/display.html')

@app.route("/display-directory")
def display_directory():
    return render_template('/display-directory.html')

@app.route("/upload-directory", methods=['POST'])
def upload_directory():
    
    # Get Pre and Post Directories
    pre = request.files['pre-directory']
    post = request.files['post-directory']

    pref = os.path.join(pre_path, "pre-directory.zip")
    postf = os.path.join(post_path, "pre-directory.zip")

    pre.save(pref)
    post.save(postf)

    # Extract Zip files
    zip_ref = zipfile.ZipFile(pref, 'r')
    zip_ref.extractall("static/upload/dir/pre")
    zip_ref.close()

    zip_ref = zipfile.ZipFile(postf, 'r')
    zip_ref.extractall("static/upload/dir/post")
    zip_ref.close()

    # Delete Zip Files
    os.remove(pref)
    os.remove(postf)

    # Open 
    for x in os.listdir('static/upload/dir/pre'):
        if os.path.isdir('static/upload/dir/pre/'+x):
            dirpre='static/upload/dir/pre/'+x
    
    for x in os.listdir('static/upload/dir/post/'):
        if os.path.isdir('static/upload/dir/post/'+x):
            dirpost='static/upload/dir/post/'+x
    
    listpre = os.listdir(dirpre)
    listpost = os.listdir(dirpost)

    distances = []

    for x in range (0,len(listpre)):
        pref = dirpre + "/" +listpre[x]
        postf = dirpost + "/" +listpost[x]

        im = Image.open(pref)
        im = im.resize((50, 50), Image.NEAREST)
        im.save(pref, "JPEG")
        im = Image.open(postf)
        im = im.resize((50, 50), Image.NEAREST)
        im.save(postf, "JPEG")

        # Load Images
        dataset = AppDataset()
        dataloader = DataLoader(dataset,
                            shuffle=False,
                            num_workers=4,
                            batch_size=1)

        data_iter = iter(dataloader)
        (img0, img1) = next(data_iter)


        # Calculate distance
        img0, img1 = Variable(img0).cuda(), Variable(img1).cuda()
        # img0 = img0.unsqueeze(0)
        (img0_output, img1_output)  = net(img0, img1)
        
        # img0_output
        # img1_output
        
        euclidean_distance = F.pairwise_distance(img0_output, img1_output)

        euclidean_distance = euclidean_distance.data.cpu().numpy()[0][0]

        distances.append(euclidean_distance)

    return str(distances[0])

@app.route("/upload", methods=['POST'])
def upload():

    # Get Pre and Post Images
    pre = request.files['pre']
    post = request.files['post']

    # Store path and name for images
    pref = os.path.join(pre_path, "PRE.jpg")
    postf = os.path.join(post_path, "POST.jpg")
    
    # Image save
    pre.save(pref)
    post.save(postf)
    
    im = Image.open(pref)
    im = im.resize((50, 50), Image.NEAREST)
    im.save(pref, "JPEG")
    im = Image.open(postf)
    im = im.resize((50, 50), Image.NEAREST)
    im.save(postf, "JPEG")

    # Load Images
    dataset = AppDataset()
    dataloader = DataLoader(dataset,
                        shuffle=False,
                        num_workers=4,
                        batch_size=1)

    data_iter = iter(dataloader)
    (img0, img1) = next(data_iter)


    # Calculate distance
    img0, img1 = Variable(img0).cuda(), Variable(img1).cuda()
    # img0 = img0.unsqueeze(0)
    output = net(img0, img1)
    
    return str(output.data.cpu().numpy()[0][0])

if __name__ == "__main__":
    app.run()