import random
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as dset
from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np

class AppDataset(Dataset):

    def __init__(self):

        self.dataset_pre = dset.ImageFolder(root="static/upload/pre")
        self.dataset_post = dset.ImageFolder(root="static/upload/post")

        self.number_of_images = len(self.dataset_pre.imgs)

    def __len__(self):
        return self.number_of_images
    
    def __getitem__(self, idx):
        
        # Anchor
        img0_tuple = self.dataset_pre.imgs[idx]
        img1_tuple = self.dataset_post.imgs[idx]
        
        assert img1_tuple[1] == img0_tuple[1]

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        
        transform=transforms.Compose([transforms.ToTensor()])

        img0 = img0.convert("L")
        img1 = img1.convert("L")

        img0 = transform(img0)
        img1 = transform(img1)

        return img0, img1
    
    
if __name__ == "__main__":
    eyc_data = AppDataset()