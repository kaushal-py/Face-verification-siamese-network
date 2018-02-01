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

class EycDataset(Dataset):
    """
    Extract and load the EYC dataset.
    Perform transformations on the dataset as required
    """

    def __init__(self, zip_path="eyc-data.tar.gz", train=False, train_size=800):
        """
        Initialisation of the dataset does the following actions - 
        1. Extract the dataset tar file.
        2. Divide the dataset into train and test set. 
        3. Augment the dataset for training purposes.
        """

        # Class states 
        self.dataset_folder_name = '.eycdata'
        self.train = train
        self.train_size = train_size

        # Check if the path to tar file is vaild
        if not os.path.isfile(zip_path):
            print("EYC dataset zip file not found. Please check for the 'tar.gz' file.")
            return
        
        # Extract the tar file
        if not os.path.isdir('.eycdata'):
            print("Extracting data from zipped file ", zip_path, "..")
            eyc_tar = tarfile.open(zip_path, "r:gz")
            eyc_tar.extractall(self.dataset_folder_name)
            print("Done extracting files to ", self.dataset_folder_name)

            # Get pre and post image files from the directory
            data_pre = sorted(os.listdir(".eycdata/pre"))
            data_post = sorted(os.listdir(".eycdata/post"))

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

            self.moveToFolder(".eycdata/pre", data_pre_train, ".eycdata/train/pre")
            self.moveToFolder(".eycdata/pre", data_pre_test, ".eycdata/test/pre")
            self.moveToFolder(".eycdata/post", data_post_train, ".eycdata/train/post")
            self.moveToFolder(".eycdata/post", data_post_test, ".eycdata/test/post")

        else:
            print("Data folder already created.")

        if self.train:
            self.augment_images(".eycdata/train/pre")
            self.augment_images(".eycdata/train/post")
        
        if self.train == True:
            self.dataset_pre = dset.ImageFolder(root=Config.training_dir_pre)
            self.dataset_post = dset.ImageFolder(root=Config.training_dir_post)
            
        else:
            self.dataset_pre = dset.ImageFolder(root=Config.testing_dir_pre)
            self.dataset_post = dset.ImageFolder(root=Config.testing_dir_post)

        self.number_of_images = len(self.dataset_pre)

    def __len__(self):
        return self.number_of_images
    
    def __getitem__(self, idx):
        
        # Anchor
        anchor_tuple = self.dataset_pre.imgs[idx]
        
        # Positive
        probability = random.randint(0, 100)
        if self.train:
            similar_idx = (idx//5 * 5) + random.randint(0, 4)
            if  probability < 0:
                positive_tuple = self.dataset_post.imgs[similar_idx]
            else:
                positive_tuple = self.dataset_pre.imgs[similar_idx]
        else:
            similar_idx = idx
            positive_tuple = self.dataset_post.imgs[similar_idx]
        
        assert anchor_tuple[1] == positive_tuple[1]

        if probability < 0:
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

        anchor = anchor.convert("L")
        positive = positive.convert("L")
        negative = negative.convert("L")

        anchor = transform(anchor)
        positive = transform(positive)
        negative = transform(negative)
        
        return anchor, positive, negative
    
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
            p.sample(5000)
        else:
            print("Augmented folder already exists at", data_folder + "/" + dest_folder)

if __name__ == "__main__":
    eyc_data = EycDataset(train=True)