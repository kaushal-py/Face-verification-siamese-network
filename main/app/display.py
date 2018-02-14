from flask import Flask, render_template, request
import os
from torch.autograd import Variable
import torch.nn.functional as F
from PIL import Image, ImageChops
import torchvision.transforms as transforms
import torch
from siameseContrastive import SiameseNetwork
from appDataset import AppDataset
from appDatasetDuplicates import AppDatasetDuplicates
from torch.utils.data import DataLoader,Dataset
import numpy as np
import zipfile
import os
import pandas as pd
import shutil
from flask_socketio import SocketIO, send

app = Flask(__name__)
app.config['CACHE_TYPE'] = "null"
app.config['SECRET_KEY'] = "kaushalrocks"
socketio = SocketIO(app)

pre_path = "static/upload/pre/temp"
post_path = "static/upload/post/temp"

# Load model
# net = torch.load('model.pt', map_location = lambda storage, loc:storage).eval()
net_pre = torch.load('model_duplicate_pre.pt', map_location = lambda storage, loc:storage).eval()
net_post = torch.load('model_duplicate_post.pt', map_location = lambda storage, loc:storage).eval()
net = torch.load('model_duplicate_pre.pt', map_location = lambda storage, loc:storage).eval()
# net_pre = torch.load('model_duplicate_pre.pt')
# net_post = torch.load('model_duplicate_post.pt')

# @app.route("/")
# def display():
#     return render_template('/display-directory.html')

@app.route("/display-single")
def display():
    return render_template('/display.html')

@app.route("/")
def display_directory():
    return render_template('/display-directory.html')

@app.route("/processing")
def processing():

    pref = os.path.join(pre_path, "pre-directory.zip")
    postf = os.path.join(post_path, "pre-directory.zip")

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
        if os.path.isdir('static/upload/dir/pre/'+str(x)):
            dirpre='static/upload/dir/pre/'+str(x)
    
    for x in os.listdir('static/upload/dir/post/'):
        if os.path.isdir('static/upload/dir/post/'+str(x)):
            dirpost='static/upload/dir/post/'+str(x)
    
    listpre = sorted(os.listdir(dirpre))
    listpost = sorted(os.listdir(dirpost))

    for x in range (0,len(listpre)):

        # Create folder and move image to that folder
        os.mkdir(dirpre+"/"+str(x))
        os.mkdir(dirpost+"/"+str(x))

        pref = dirpre + "/" + str(x) + "/" + listpre[x]
        postf = dirpost + "/" + str(x) + "/" + listpost[x]

        os.rename(dirpre+"/"+listpre[x],pref)
        os.rename(dirpost+"/"+listpost[x],postf)

    # Load Images
    dataset_pre = AppDatasetDuplicates(dirpre,dirpre,False)
    dataloader_pre = DataLoader(dataset_pre,
                        shuffle=False,
                        num_workers=4,
                        batch_size=1)

    dataset_post = AppDatasetDuplicates(dirpost,dirpost,False)
    dataloader_post = DataLoader(dataset_post,
                        shuffle=False,
                        num_workers=4,
                        batch_size=1)

    dataset = AppDatasetDuplicates(dirpre,dirpost,False)
    dataloader = DataLoader(dataset,
                        shuffle=False,
                        num_workers=4,
                        batch_size=1)

    match_dist = []
    match_pre = []
    match_post = []
    pre_match_img = []
    pre_match_dist = []
    pre_match_old = []
    post_match_img = []
    post_match_dist = []
    post_match_old = []

    cnt_pre_cnt = []
    cnt_pre_name = []
    cnt_post_cnt = []
    cnt_post_name = []


    data_iter_pre = iter(dataloader_pre)
    data_iter_post = iter(dataloader_post)
    data_iter = iter(dataloader)

    img_output_pre = []
    img_output_post = []

    for x in range(len(listpre)):

        cnt_post, cnt_pre = checkPhotoshop(dirpre + "/" + str(x) + "/" , listpre[x],dirpost + "/" + str(x) + "/" , listpost[x])
        if cnt_pre > 100:
            cnt_pre_cnt.append(cnt_pre)
            cnt_pre_name.append(listpre[x][:24])

        if cnt_post > 100:
            cnt_post_cnt.append(cnt_post)
            cnt_post_name.append(listpost[x][:24])


        (img0, img1) = next(data_iter_pre)

        # Calculate distance
        # img0, img1 = Variable(img0).cuda(), Variable(img1).cuda()
        img0, img1 = Variable(img0), Variable(img1)
        (img0_output_pre, img1_output)  = net_pre(img0, img1)
        img1_output = img1_output.data.numpy()[0]
        output_pre = str(img1_output.tolist())[1:-1]
        csv_pre = open("static/pre_values.csv",'a')
        csv_pre.write("\nimages/pre/"+listpre[x]+", "+str(output_pre))
        csv_pre.close()

        img0_output_pre = img0_output_pre.data.numpy()[0]
        img_output_pre.append(img0_output_pre)


        (img0, img1) = next(data_iter_post)

        # Calculate distance
        img0, img1 = Variable(img0), Variable(img1)
        # img0, img1 = Variable(img0).cuda(), Variable(img1).cuda()
        (img0_output_post, img1_output)  = net_post(img0, img1)
        img1_output = img1_output.data.numpy()[0]
        output_post = str(img1_output.tolist())[1:-1]
        csv_post = open("static/post_values.csv",'a')
        csv_post.write("\nimages/post/"+listpost[x]+", "+output_post)
        csv_post.close()

        img0_output_post = img0_output_post.data.numpy()[0]
        img_output_post.append(img0_output_post)

        (img0, img1) = next(data_iter)

        # Calculate distance
        # img0, img1 = Variable(img0).cuda(), Variable(img1).cuda()
        img0, img1 = Variable(img0), Variable(img1)
        # img0 = img0.unsqueeze(0)
        (img0_output, img1_output)  = net(img0, img1)
        
        euclidean_distance = F.pairwise_distance(img0_output, img1_output)

        euclidean_distance = euclidean_distance.data.cpu().numpy()[0][0]

        match_pre.append("/images/pre/"+listpre[x])
        match_post.append("/images/post/"+listpost[x])
        match_dist.append(euclidean_distance)
    
    # Read csv here
    df_pre = pd.read_csv("static/pre_values.csv",delimiter=', ')
    df_post = pd.read_csv("static/post_values.csv",delimiter=', ')

    for x in range(0,len(img_output_pre)):
        for y in range(0,df_pre.count()[0]-1):
            get_val = df_pre.iloc[y][1:129].as_matrix(columns=None)
            euclidean_distance = np.linalg.norm(img_output_pre[x] - get_val)
            if euclidean_distance < 0.5:
                pre_match_dist.append(euclidean_distance)
                pre_match_img.append("/images/pre/"+listpre[x])
                pre_match_old.append(df_pre.iloc[y][0])
            else :
                csv_img = open("static/images.csv",'a')
                csv_img.write("\n/images/pre/"+listpre[x]+", "+str(1))
                csv_img.close()
    
    for x in range(0,len(img_output_post)):
        for y in range(0,df_post.count()[0]-1):
            get_val = df_post.iloc[y][1:129].as_matrix(columns=None)
            euclidean_distance = np.linalg.norm(img_output_post[x] - get_val)
            if euclidean_distance < 1:
                post_match_dist.append(euclidean_distance)
                post_match_img.append("/images/post/"+listpost[x])
                post_match_old.append(df_post.iloc[y][0])
            else :
                csv_img = open("static/images.csv",'a')
                csv_img.write("\n/images/post/"+listpre[x]+", "+str(1))
                csv_img.close()

    for x in range(0,len(listpre)):
        os.rename(dirpre + "/" + str(x) + "/" + listpre[x],"static/images/pre/"+listpre[x])
        os.rename(dirpost + "/" + str(x) + "/" + listpost[x],"static/images/post/"+listpost[x])

        shutil.rmtree(dirpre + "/" + str(x))
        shutil.rmtree(dirpost + "/" + str(x))

    match = [match_dist, match_pre, match_post] 
    pre_match = [pre_match_dist,pre_match_img,pre_match_old]
    post_match = [post_match_dist,post_match_img,post_match_old]
    match_len = len(match[0])
    pre_len = len(pre_match[0])
    post_len = len(post_match[0])
    cnt_pre_list = [cnt_pre_name, cnt_pre_cnt]
    cnt_post_list = [cnt_post_name, cnt_post_cnt]
    shutil.rmtree(dirpre)
    shutil.rmtree(dirpost)
    return render_template("result-directory.html", pre_match = pre_match, pre_len = pre_len, post_match = post_match, post_len = post_len, match = match, match_len = match_len, cnt_pre_list = cnt_pre_list, cnt_post_list = cnt_post_list, ps_pre = cnt_pre_name, ps_post = cnt_post_name)

@app.route("/upload-directory", methods=['POST'])
def upload_directory():
    
    # Get Pre and Post Directories
    pre = request.files['pre-directory']
    post = request.files['post-directory']

    pref = os.path.join(pre_path, "pre-directory.zip")
    postf = os.path.join(post_path, "pre-directory.zip")

    pre.save(pref)
    post.save(postf)

    return render_template("processing.html")

    

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
    
    # Check if image is Photoshopped
    cnt_post, cnt_pre = checkPhotoshop()

    # Load Images
    dataset = AppDataset()
    dataloader = DataLoader(dataset,
                        shuffle=False,
                        num_workers=4,
                        batch_size=1)

    data_iter = iter(dataloader)
    (img0, img1) = next(data_iter)


    # Calculate distance
    # img0, img1 = Variable(img0).cuda(), Variable(img1).cuda()
    img0, img1 = Variable(img0), Variable(img1)
    # img0 = img0.unsqueeze(0)
    (img0_output, img1_output)  = net(img0, img1)
        
    df_pre = pd.read_csv("static/pre_values.csv",delimiter=', ')
    df_post = pd.read_csv("static/post_values.csv",delimiter=', ')

    img1_output_post = img1_output.data.numpy()[0]
    img0_output_pre = img0_output.data.numpy()[0]

    for y in range(0,df_post.count()[0]-1):
        get_val_pre = df_pre.iloc[y][1:129].as_matrix(columns=None)
        get_val_post = df_post.iloc[y][1:129].as_matrix(columns=None)
        euclidean_distance_pre = np.linalg.norm(img0_output_pre - get_val_pre)
        euclidean_distance_post = np.linalg.norm(img1_output_post - get_val_post)
    
    euclidean_distance = F.pairwise_distance(img0_output, img1_output)

    euclidean_distance = euclidean_distance.data.cpu().numpy()[0][0]

    # distances.append(euclidean_distance)
    print(euclidean_distance)

    return render_template("result-single.html", distance = euclidean_distance, cnt_post = cnt_post, cnt_pre=cnt_pre)


@app.route("/rejected-list")
def fetchRejected():
    df = pd.read_csv("static/images.csv", delimiter=', ')
    rejected_list = []
    for row in df.itertuples():
        if row.status == 0:
            rejected_list.append(row.path)
    return render_template("rejected-list.html", rejected_list = rejected_list)


@socketio.on('reject')
def handleRejected(img):
    print("The index is "+str(img))
    csv_img = open("static/images.csv",'a')
    csv_img.write("\n"+str(img)+", "+str(0))
    csv_img.close()

@socketio.on('accept')
def handleRejected(img):
    print("The index is "+str(img))
    csv_img = open("static/images.csv",'a')
    csv_img.write("\n"+str(img)+", "+str(1))
    csv_img.close()


def checkPhotoshop(pre_path, pre_name, post_path, post_name):

    ORIG_POST = os.path.join(post_path, post_name)
    TEMP_POST = os.path.join(post_path, "POST-TEMP.jpg")
    ORIG_PRE = os.path.join(pre_path, pre_name)
    TEMP_PRE = os.path.join(pre_path, "PRE-TEMP.jpg")
    SCALE = 15

    original_post = Image.open(ORIG_POST)
    original_post.save(TEMP_POST, quality=90)
    temporary_post = Image.open(TEMP_POST)
    original_pre = Image.open(ORIG_PRE)
    original_pre.save(TEMP_PRE, quality=90)
    temporary_pre = Image.open(TEMP_PRE)
    
    diff_post = ImageChops.difference(original_post, temporary_post)
    d_post = diff_post.load()
    WIDTH, HEIGHT = diff_post.size
    for x in range(WIDTH):
        for y in range(HEIGHT):
            d_post[x, y] = tuple(k * SCALE for k in d_post[x, y])

    diff_post.save(post_path+"/POST-ELA.jpg")

    diff_pre = ImageChops.difference(original_pre, temporary_pre)
    d_pre = diff_pre.load()
    WIDTH, HEIGHT = diff_pre.size
    for x in range(WIDTH):
        for y in range(HEIGHT):
            d_pre[x, y] = tuple(k * SCALE for k in d_pre[x, y])

    diff_pre.save(pre_path+"/PRE-ELA.jpg")
    
    # Count the number of pixels in ELA image
    img_ela_post = np.asarray(Image.open(post_path+"/POST-ELA.jpg"))
    img_hist_post = np.histogram(img_ela_post.ravel(),256,[0,256])
    cnt_post = np.count_nonzero(img_hist_post[0])

    img_ela_pre = np.asarray(Image.open(pre_path+"/PRE-ELA.jpg"))
    img_hist_pre = np.histogram(img_ela_pre.ravel(),256,[0,256])
    cnt_pre = np.count_nonzero(img_hist_pre[0])
    
    return cnt_post, cnt_pre

if __name__ == "__main__":
    # app.run(debug=True)
    socketio.run(app, debug=True)


# df = pd.read_csv("pre_values.csv")
# for x in range(0,df.count()[0]-1):
#     get_val1 = df.iloc[x][1:129].as_matrix(columns=None)
#     img1_path = df.iloc[x][0].as_matrix(columns=None)
#     for y in range(0,df.count()[0]-1):
#             get_val2 = df.iloc[y][1:129].as_matrix(columns=None)            
#             img2_path = df.iloc[y][0].as_matrix(columns=None)
#             euclidean_distance = np.linalg.norm(get_val1 - get_val2)    