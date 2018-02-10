import os
import random
import tarfile
import shutil
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as dset
import Augmentor
from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np

class EycDataset(Dataset):
    """
    Extract and load the EYC dataset.
    Perform transformations on the dataset as required
    """

    def __init__(self, comparison, zip_path="eycdata.tar.gz"):
        """
        Initialisation of the dataset does the following actions - 
        1. Extract the dataset tar file.
        2. Divide the dataset into train and test set. 
        3. Augment the dataset for training purposes.
        """

        # Class states 
        self.dataset_folder_name = 'static/eycdata'
        self.train = False

        self.comparison = comparison

        # Check if the path to tar file is vaild
        if not os.path.isfile(zip_path):
            print("EYC dataset zip file not found. Please check for the 'tar.gz' file.")
            return
        
        # Extract the tar file
        if not os.path.isdir('static/eycdata'):
            print("Extracting data from zipped file ", zip_path, "..")
            eyc_tar = tarfile.open(zip_path, "r:gz")
            eyc_tar.extractall(self.dataset_folder_name)
            print("Done extracting files to ", self.dataset_folder_name)

        else:
            print("Data folder already created.")

        self.augment_images('static/eycdata/pre', dest_folder='../augmented/pre')
        self.augment_images('static/eycdata/post', dest_folder='../augmented/post')

        self.dataset_pre = dset.ImageFolder(root="static/eycdata/pre")
        self.dataset_post = dset.ImageFolder(root="static/eycdata/post")

        self.number_of_images = len(self.dataset_pre.imgs)

    def __len__(self):
        return self.number_of_images
    
    def __getitem__(self, idx):
        
        if self.comparison=="pre-post":
            probability = random.randint(0, 100)
        elif self.comparison=="pre-pre":
            probability = 0
        else:
            probability = 100
        
        if self.train:
            similar_idx = (idx//20 * 20) + random.randint(0, 19)
        else:
            similar_idx = (idx//10 * 10) + random.randint(0, 9)

        if  probability < 50:
            anchor_tuple = self.dataset_pre.imgs[idx]
            
            if self.comparison=="pre-post":
                positive_tuple = self.dataset_post.imgs[similar_idx]
            else:
                positive_tuple = self.dataset_pre.imgs[similar_idx]
        
        else:
            anchor_tuple = self.dataset_post.imgs[idx]
            
            if self.comparison=="pre-post":
                positive_tuple = self.dataset_pre.imgs[similar_idx]
            else:
                positive_tuple = self.dataset_post.imgs[similar_idx]
    
        assert anchor_tuple[1] == positive_tuple[1]

        if probability < 50:
            while True:
                negative_tuple = random.choice(self.dataset_pre.imgs)
                if negative_tuple[1] != anchor_tuple[1]:
                    break
        else:
            while True:
                negative_tuple = random.choice(self.dataset_post.imgs)
                if anchor_tuple[1] != negative_tuple[1]:
                    break

        anchor = Image.open(anchor_tuple[0])
        positive = Image.open(positive_tuple[0])
        negative = Image.open(negative_tuple[0])
        
        transform=transforms.Compose([transforms.ToTensor()])

        anchor = anchor.resize((50, 50), Image.ANTIALIAS)
        positive = positive.resize((50, 50), Image.ANTIALIAS)
        negative = negative.resize((50, 50), Image.ANTIALIAS)

        # anchor = anchor.convert("L")
        # positive = positive.convert("L")

        anchor = transform(anchor)
        positive = transform(positive)
        negative = transform(negative)
        
        return anchor_tuple[0], positive_tuple[0], negative_tuple[0], anchor, positive, negative

    
    def augment_images(self, data_folder, dest_folder="augmented"):
        '''
        If train cycle, augment the images.
        Transformations done - 
        1. Flip left-right (5o% probability)
        2. Rotate (70% probability, angles -5 -> +5)
        3. Zoom (30% probability, max_factor = 1.2)
        '''

        if not os.path.exists(os.path.join(data_folder, dest_folder)):
            print("== Augmenting images at", data_folder, " ==")
            p = Augmentor.Pipeline(data_folder, output_directory=dest_folder)
            p.flip_left_right(probability=0.5)
            p.rotate(probability=0.7, max_left_rotation=15, max_right_rotation=15)
            p.zoom(probability=0.3, min_factor=1, max_factor=1.3)
            p.random_distortion(probability=0.3, grid_width=4, grid_height=4, magnitude=1)
            p.sample(10000)
            # p.resize(probability=1, height=100, width=100)
        else:
            print("Augmented folder already exists at", data_folder + "/" + dest_folder)

    
    
if __name__ == "__main__":
    eyc_data = EycDataset(comparison="pre-post")