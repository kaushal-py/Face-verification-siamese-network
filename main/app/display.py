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

app = Flask(__name__)

pre_path = "static/upload/pre/temp"
post_path = "static/upload/post/temp"

# Load model
net = torch.load('../model.pt').eval()

@app.route("/")
def hello():
    return render_template('/display.html')

@app.route('/upload', methods=['POST'])
def upload_pre():

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