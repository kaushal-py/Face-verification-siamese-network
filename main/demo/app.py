from flask import Flask, render_template, request
from flask_socketio import SocketIO

from eyc_dataset_loader import EycDataset
from torch.utils.data import DataLoader,Dataset
import torch
from torch.autograd import Variable
import torch.nn.functional as F

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

net = torch.load('../models/model_triplet_pr_pr3.pt')

@app.route('/intro')
def intro():

    # 0.0.0.0:5000/intro?size=60&images=72&time=0
    # 0.0.0.0:5000/intro?size=120&images=25&time=1
    #120, 25
    #60 , 72
    size = int(request.args.get('size'))
    images = int(request.args.get('images'))
    wait_time = int(request.args.get('time'))
    
    _thread.start_new_thread(pre_post_comparisons, (wait_time/2,))
    
    return render_template("intro.html", size=size, images=images)
    
def pre_post_omparisons(wait_time):
    
    data_iter = iter(dataloader_pre)

    while True:

        anchor_tuple, positive_tuple, negative_tuple, anchor, positive, negative = next(data_iter)

        anchor_in, positive_in, negative_in = Variable(anchor).cuda(), Variable(positive).cuda() , Variable(negative).cuda()
        (anchor_output, positive_output, negative_output)  = net(anchor_in, positive_in, negative_in)
        
        same_distance = F.pairwise_distance(anchor_output, positive_output)
        diff_distance = F.pairwise_distance(anchor_output, negative_output)

        same_distance = same_distance.data.cpu().numpy()[0][0]
        diff_distance = diff_distance.data.cpu().numpy()[0][0]

        # print(same_distance, diff_distance)
        
        # print(anchor_tuple)
        probability = random.randint(0, 1)
        if probability == 0:
            if same_distance < 0.5:
                color = "success"      
            else:
                color = "warning"
            socketio.emit('pre', {'color': color,
                                  'img0': anchor_tuple, 
                                  'img1': positive_tuple})
        else:
            if diff_distance > 0.5:
                color = "warning"      
            else:
                color = "success"
            socketio.emit('pre', {'color': color,
                                  'img0': anchor_tuple, 
                                  'img1': negative_tuple})  

        time.sleep(wait_time)      

@app.route('/vector')
def display_vector():
    return render_template('vector.html')

@app.route('/')
def homepage():    
    image_set = []
    dataiter = iter(dataloader_pre)

    for _ in range(6):
        image_set.append(next(dataiter))
    
    return render_template("main.html", images=image_set)

# def homepage_callback():
    
#     # time.sleep(1)
#     dataiter = iter(dataloader)
#     for _ in range(100):
#         img0 = next(dataiter)[0][0]
#         # print(img0)
#         socketio.emit('counter', {'data': img0})
#         time.sleep(2)

@app.route('/photoshop')
def photoshoppage():
    return render_template("photoshop.html")

@app.route('/augmentation')
def augmentpage():
    class_set = sorted(os.listdir('static/eycdata/pre/augmented/'))
    # print(img_set)
    img_class = random.choice(class_set)
    image_set = sorted(os.listdir('static/eycdata/pre/augmented/'+img_class))

    image_set = [ 'static/eycdata/pre/augmented/'+img_class+'/'+image for image in image_set]
    return render_template("augment.html", images=image_set)

if __name__ == "__main__":
    socketio.run(app, host='0.0.0.0', debug=True)
