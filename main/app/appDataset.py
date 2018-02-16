import random
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as dset
from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np
from eye_detector import Preprocess

class AppDataset(Dataset):

    def __init__(self):

        self.dataset_pre = dset.ImageFolder(root="static/upload/pre")
        self.dataset_post = dset.ImageFolder(root="static/upload/post")

        self.p = Preprocess("static/haarcascades_eye.xml", "static/patch.jpg")

        self.number_of_images = len(self.dataset_pre.imgs)

    def __len__(self):
        return self.number_of_images
    
    def __getitem__(self, idx):
        
        # Anchor
        anchor_tuple = self.dataset_pre.imgs[idx]
        positive_tuple = self.dataset_post.imgs[idx]
        
        anchor = self.p.subtract_backgroud(anchor_tuple[0])
        positive = self.p.subtract_backgroud(positive_tuple[0])
        # positive = self.p.add_eyeptach(positive)

        anchor = cv2.cvtColor(anchor,cv2.COLOR_BGR2RGB)
        positive = cv2.cvtColor(positive,cv2.COLOR_BGR2RGB)
        
        anchor = Image.fromarray(anchor)
        positive = Image.fromarray(positive)

        transform=transforms.Compose([transforms.ToTensor()])

        anchor = anchor.convert("L")
        positive = positive.convert("L")

        # anchor = anchor.resize((216, 216), Image.ANTIALIAS)
        # positive = positive.resize((216, 216), Image.ANTIALIAS)
        # negative = negative.resize((216, 216), Image.ANTIALIAS)

        anchor = transform(anchor)
        positive = transform(positive)
        
        return anchor, positive
    
    
if __name__ == "__main__":
    eyc_data = AppDataset()