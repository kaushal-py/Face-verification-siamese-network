import os
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as dset
from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np

class AppDatasetDuplicates(Dataset):
    """
    Extract and load the EYC dataset.
    Perform transformations on the dataset as required
    """

    def __init__(self, pre, post, train=False, train_size=200):

        # Class states 
        self.pre = pre
        self.post = post

        self.dataset_pre = dset.ImageFolder(root=pre)
        self.dataset_post = dset.ImageFolder(root=post)

        self.number_of_images = len(self.dataset_pre.imgs)

    def __len__(self):
        return self.number_of_images
    
    def __getitem__(self, idx):

        img0_tuple = self.dataset_pre.imgs[idx]
        img1_tuple = self.dataset_pre.imgs[idx]

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        
        transform=transforms.Compose([transforms.ToTensor()])

        img0 = img0.convert("L")
        img1 = img1.convert("L")

        img0 = transform(img0)
        img1 = transform(img1)
                
        return img0, img1

if __name__ == "__main__":
    eyc_data = EycDataset(train=True)
