from flask import Flask, render_template
from eyc_dataset_loader import EycDataset
from torch.utils.data import DataLoader,Dataset
import os
import random

app = Flask(__name__)

dataset = EycDataset()

dataloader = DataLoader(dataset,
                        shuffle=True,
                        num_workers=8,
                        batch_size=1)

@app.route('/')
def homepage():    
    image_set = []
    dataiter = iter(dataloader)

    for _ in range(6):
        image_set.append(next(dataiter))
    
    return render_template("main.html", images=image_set)

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
    app.run(host='0.0.0.0')
