import os
import random
import tarfile
import shutil
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as dset
import Augmentor
from PIL import Image
from PIL import ImageOps
import torchvision.transforms as transforms
from config import Config

class EycDataset(Dataset):
    """
    Extract and load the EYC dataset.
    Perform transformations on the dataset as required
    """

    def __init__(self, csv_path="static/pre_values.csv"):
        
        self.csv_path = csv_path

        self.csv = open(self.csv_path,'a')

        self.number_of_images = len(self.dataset_pre)

    def __len__(self):
        
        return self.number_of_images
    
    def __getitem__(self, idx):
        
        return anchor, positive, negative
    
if __name__ == "__main__":
    eyc_data = EycDataset()