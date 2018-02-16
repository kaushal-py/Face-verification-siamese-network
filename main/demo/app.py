from flask import Flask, render_template, request
from flask_socketio import SocketIO

from eyc_dataset_loader import EycDataset
from torch.utils.data import DataLoader,Dataset
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from PIL import ImageOps

import os
import random
import time
import _thread

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

dataset_pre = EycDataset("pre-pre")
dataset_pre_post = EycDataset("pre-post")
dataset_post = EycDataset("post-post")

dataloader_pre = DataLoader(dataset_pre,
                        shuffle=True,
                        num_workers=8,
                        batch_size=1)
dataloader_post = DataLoader(dataset_post,
                        shuffle=True,
                        num_workers=8,
                        batch_size=1)
dataloader_pre_post = DataLoader(dataset_pre_post,
                        shuffle=True,
                        num_workers=8,
                        batch_size=1)

print("Loading models.. ")
net_pre = torch.load('../models/model_triplet_pr_pr3.pt').eval()
net_pre_post = torch.load('../models/model_triplet_pr_po_max_pool_large_600.pt').eval()
print("Models loaded")

@app.route('/')
def intro():

    # http://0.0.0.0:5000/intro?size=60&images=72&time=0
    # http://0.0.0.0:5000/intro?size=120&images=25&time=1
    #120, 25
    #60 , 72
    # size = int(request.args.get('size'))
    # images = int(request.args.get('images'))
    # wait_time = float(request.args.get('time'))
    
    size = 67
    images = 56
    wait_time = 0.05
    
    _thread.start_new_thread(pre_post_comparisons, (wait_time,))
    
    return render_template("intro.html", size=size, images=images)

def pre_post_comparisons(wait_time):
    
    while(True):
        data_iter = iter(dataloader_pre_post)

        for _ in range(1000):

            anchor_tuple, positive_tuple, negative_tuple, anchor, positive, negative = next(data_iter)

            anchor_in, positive_in, negative_in = Variable(anchor).cuda(), Variable(positive).cuda() , Variable(negative).cuda()
            (anchor_output, positive_output, negative_output)  = net_pre_post(anchor_in, positive_in, negative_in)
            
            same_distance = F.pairwise_distance(anchor_output, positive_output)
            diff_distance = F.pairwise_distance(anchor_output, negative_output)

            same_distance = same_distance.data.cpu().numpy()[0][0]
            diff_distance = diff_distance.data.cpu().numpy()[0][0]

            # print(same_distance, diff_distance)
            
            # print(anchor_tuple)
            probability = 0
            if probability == 0:
                if same_distance < 0.9:
                    color = "green"      
                else:
                    color = "danger"
                socketio.emit('pre', {'color': color,
                                    'img0': anchor_tuple, 
                                    'img1': positive_tuple})
            else:
                if diff_distance > 0.9:
                    color = "danger"      
                else:
                    color = "green"
                socketio.emit('pre', {'color': color,
                                    'img0': anchor_tuple, 
                                    'img1': negative_tuple})  

            time.sleep(wait_time)      

@app.route('/rcpr')
def rcpr():
    return render_template("rcpr.html", images=["static/1.JPG", "static/2.JPG"])

@app.route('/vector')
def display_vector():

    _thread.start_new_thread(vector, ())
    return render_template('vector.html')

def vector():
    data_iter = iter(dataloader_pre_post)

    while True:

        anchor_tuple, positive_tuple, _, anchor, positive, negative = next(data_iter)

        anchor_in, positive_in, negative_in = Variable(anchor).cuda(), Variable(positive).cuda() , Variable(negative).cuda()
        (anchor_output, positive_output, _)  = net_pre_post(anchor_in, positive_in, negative_in)

        same_distance = F.pairwise_distance(anchor_output, positive_output)
         
        anchor_output = anchor_output.data.cpu().numpy()[0]
        anchor_output = (np.array2string(anchor_output, precision=3, separator='\t,\t', suppress_small=True))

        positive_output = positive_output.data.cpu().numpy()[0]
        positive_output = (np.array2string(positive_output, precision=3, separator='\t,\t', suppress_small=True))

        same_distance = str(same_distance.data.cpu().numpy()[0][0])
        same_distance = same_distance[:4]
        # output = str(anchor_output[:3]) + " ...." + str(anchor_output[-3:-1])
        time.sleep(3)

        socketio.emit('vector', {'img_a' : anchor_tuple,
                    'img_p' : positive_tuple, 
                    'vector_a' : str(anchor_output),
                    'vector_p' : str(positive_output),
                    'distance' : same_distance})

@app.route('/triplet')
def display_triplet():

    _thread.start_new_thread(triplet, ())
    return render_template('triplet.html')

def triplet():
    data_iter = iter(dataloader_pre_post)

    while True:

        anchor_tuple, positive_tuple, negative_tuple, anchor, positive, negative = next(data_iter)

        anchor_in, positive_in, negative_in = Variable(anchor).cuda(), Variable(positive).cuda() , Variable(negative).cuda()
        (anchor_output, positive_output, negative_output)  = net_pre_post(anchor_in, positive_in, negative_in)

        same_distance = F.pairwise_distance(anchor_output, positive_output)
        diff_distance = F.pairwise_distance(anchor_output, negative_output)
         
        same_distance = str(same_distance.data.cpu().numpy()[0][0])
        same_distance = same_distance[:4]

        diff_distance = str(diff_distance.data.cpu().numpy()[0][0])
        diff_distance = diff_distance[:4]

        time.sleep(3)

        socketio.emit('vector', {'img_a' : anchor_tuple,
                    'img_p' : positive_tuple, 
                    'img_n' : negative_tuple, 
                    'same_distance' : same_distance,
                    'diff_distance' : diff_distance})



@app.route('/intro')
def homepage():    
    image_set = []
    dataiter = iter(dataloader_pre_post)

    for _ in range(4):
        image_set.append(next(dataiter))
    
    return render_template("main.html", images=image_set)

@app.route('/photoshop')
def photoshoppage():
    out = int(request.args.get('out'))
    return render_template("photoshop.html", out=out)

@app.route('/augmentation')
def augmentpage():
    class_set = sorted(os.listdir('static/eycdata/augmented/post/'))
    # print(img_set)
    img_class = random.choice(class_set)
    image_set = sorted(os.listdir('static/eycdata/augmented/post/'+img_class))

    image_set = [ 'static/eycdata/augmented/post/'+img_class+'/'+image for image in image_set]
    return render_template("augment.html", images=image_set)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True)
