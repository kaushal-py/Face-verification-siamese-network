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

    def __init__(self, zip_path="eyc-data.tar.gz"):
        """
        Initialisation of the dataset does the following actions - 
        1. Extract the dataset tar file.
        2. Divide the dataset into train and test set. 
        3. Augment the dataset for training purposes.
        """

        # Class states 
        self.dataset_folder_name = 'static/eycdata'

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

    
        self.dataset_pre = dset.ImageFolder(root="static/eycdata/pre")
        self.dataset_post = dset.ImageFolder(root="static/eycdata/post")

        self.number_of_images = len(self.dataset_pre.imgs)
        print(self.number_of_images)

    def __len__(self):
        return self.number_of_images
    
    def __getitem__(self, idx):
        
        # Anchor
        img0_tuple = self.dataset_pre.imgs[idx]
        
        # Positive
        label = random.randint(0, 1)
        
        probability = random.randint(1, 100)
        if label == 0:
            
            img1_tuple = self.dataset_post.imgs[idx]
            
            assert img1_tuple[1] == img0_tuple[1]
        else:
            if probability<60:
                while True:
                    img1_tuple = random.choice(self.dataset_pre.imgs)
                    if img0_tuple[1] != img1_tuple[1]:
                        break
            else:
                while True:
                    img1_tuple = random.choice(self.dataset_post.imgs)
                    if img0_tuple[1] != img1_tuple[1]:
                        break

        
        return img0_tuple, img1_tuple, label
    
    
if __name__ == "__main__":
    eyc_data = EycDataset()