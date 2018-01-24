import os
import random
import tarfile
import shutil
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as dset
import Augmentor
from PIL import Image
import torchvision.transforms as transforms
from config import Config
import torch
import numpy as np

class EycDataset(Dataset):
    """
    Extract and load the EYC dataset.
    Perform transformations on the dataset as required
    """

    def __init__(self, train=False, train_size=800):
        """
        Initialisation of the dataset does the following actions - 
        1. Extract the dataset tar file.
        2. Divide the dataset into train and test set. 
        3. Augment the dataset for training purposes.
        """

        # Class states 
        self.dataset_folder_name = '225-images'
        self.train = train
        self.train_size = train_size

        # Extract the tar file
        if not os.path.isdir('225-images/train'):

            # Get pre and post image files from the directory
            data_pre = sorted(os.listdir("225-images/pre"))
            data_post = sorted(os.listdir("225-images/post"))

            # Randomize the data
            data_pair = list(zip(data_pre, data_post))
            random.shuffle(data_pair)
            data_pre, data_post = zip(*data_pair)

            # Split into training and testing sets
            data_pre_train = data_pre[:self.train_size]
            data_pre_test = data_pre[self.train_size:]
            data_post_train = data_post[:self.train_size]
            data_post_test = data_post[self.train_size:]

            print("Making training and test data..")

            self.moveToFolder("225-images/pre", data_pre_train, "225-images/train/pre")
            self.moveToFolder("225-images/pre", data_pre_test, "225-images/test/pre")
            self.moveToFolder("225-images/post", data_post_train, "225-images/train/post")
            self.moveToFolder("225-images/post", data_post_test, "225-images/test/post")

        else:
            print("Data folder already created.")

        if self.train:
            self.augment_images("225-images/train/pre")
            self.augment_images("225-images/train/post")

        if self.train:
            self.dataset_pre = dset.ImageFolder(root="225-images/train/pre/augmented")
            self.dataset_post = dset.ImageFolder(root="225-images/train/post/augmented")
        else:
            self.dataset_pre = dset.ImageFolder(root="225-images/test/post")
            self.dataset_post = dset.ImageFolder(root="225-images/test/post")

        self.number_of_images = len(self.dataset_pre.imgs)

    def __len__(self):
        return self.number_of_images
    
    def __getitem__(self, idx):
        
        # Anchor
        img0_tuple = self.dataset_pre.imgs[idx]
        
        # Positive
        if self.train:
            label = random.randint(0, 1)
        else:
            label = 0
        probability = random.randint(1, 100)
        if label == 0:
            if self.train:
                similar_idx = (idx//5 * 5) + random.randint(0, 4)
                if  probability < 80:
                    img1_tuple = self.dataset_post.imgs[similar_idx]
                else:
                    img1_tuple = self.dataset_pre.imgs[similar_idx]
            else:
                similar_idx = idx
                img1_tuple = self.dataset_post.imgs[similar_idx]
            
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

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        
        transform=transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


        # img0 = img0.convert("L")
        # img1 = img1.convert("L")

        img0 = transform(img0)
        img1 = transform(img1)
        
        if self.train:
            return img0, img1, torch.from_numpy(np.array([int(label)],dtype=np.float32))
        else:
            return img0, img1, label
    
    def moveToFolder(self, src_folder, src_list, dest_folder):
        '''
        Move a list of folders to a destination folder
        '''
        for src_class in src_list:
            
            src_path = os.path.join(src_folder, src_class)
            dest_path = os.path.join(dest_folder, src_class)

            # class_files = os.listdir(src_path)

            shutil.copytree(src_path, dest_path)

            # for file_name in class_files:
            #     file_path = os.path.join(src_path, file_name)
            #     shutil.copy(file_path, dest_path)


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
            p.rotate(probability=0.7, max_left_rotation=5, max_right_rotation=5)
            p.zoom(probability=0.3, min_factor=1, max_factor=1.2)
            p.sample(4000)
        else:
            print("Augmented folder already exists at", data_folder + "/" + dest_folder)

if __name__ == "__main__":
    eyc_data = EycDataset(train=True)