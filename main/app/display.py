from flask import Flask, render_template, request
import os
from torch.autograd import Variable
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import torch
from siameseContrastive import SiameseNetwork

app = Flask(__name__)

UPLOAD_FOLDER = os.path.basename('upload')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model
net = torch.load('../model_improved.pt')

@app.route("/")
def hello():
    return render_template('/display.html')

@app.route('/upload', methods=['POST'])
def upload_pre():

    # Get Pre and Post Images
    pre = request.files['pre']
    post = request.files['post']

    # Store path and name for images
    pref = os.path.join(app.config['UPLOAD_FOLDER'], "PRE.jpg")
    postf = os.path.join(app.config['UPLOAD_FOLDER'], "POST.jpg")
    
    # Image save
    pre.save(pref)
    post.save(postf)
    
    # Load and preprocess images
    img0 = Image.open('upload/PRE.jpg')
    img1 = Image.open('upload/POST.jpg')

    print(img0)

    img0 = img0.resize((50, 50), Image.NEAREST)
    img1 = img1.resize((50, 50), Image.NEAREST)

    print(img0)
    
    transform=transforms.Compose([transforms.ToTensor()])

    img0 = img0.convert("L")
    img1 = img1.convert("L")

    img0 = transform(img0)
    img1 = transform(img1)

    # Calculate distance
    img0, img1 = Variable(img0).cuda(), Variable(img1).cuda()
    img0 = img0.unsqueeze(0)
    print(img0)
    (img0_output, img1_output)  = net(img0, img1)
    
    print("output")
    euclidean_distance = F.pairwise_distance(img0_output, img1_output)

    euclidean_distance = euclidean_distance.data.cpu().numpy()[0][0]

    return euclidean_distance

if __name__ == "__main__":
    app.run()