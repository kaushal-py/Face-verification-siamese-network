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
from torch import nn
from torch import optim
from torch.autograd import Variable

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

        if self.train:
            self.dataset_pre = dset.ImageFolder(root=Config.training_dir_pre)
            self.dataset_post = dset.ImageFolder(root=Config.training_dir_post)
        else:
            self.dataset_pre = dset.ImageFolder(root=Config.testing_dir_pre)
            self.dataset_post = dset.ImageFolder(root=Config.testing_dir_post)

        self.number_of_images = len(self.dataset_pre.imgs)

    def __len__(self):
        return self.number_of_images
    
    def __getitem__(self, idx):
        
        # Anchor
        label = random.randint(0, 1)

        if label == 0:
            img_tuple = self.dataset_pre.imgs[idx]
        else:
            img_tuple = self.dataset_post.imgs[idx]            
        
        img = Image.open(img_tuple[0])
        
        transform=transforms.Compose([transforms.ToTensor()])

        img = img.convert("L")

        img = transform(img)
        
        
        return img, label
    
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
            p.sample(000)
        else:
            print("Augmented folder already exists at", data_folder + "/" + dest_folder)

class ClassifierNetwork(nn.Module):
    def __init__(self):
        super(ClassifierNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),
            nn.Dropout2d(p=.2),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            nn.Dropout2d(p=.2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 16, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.Dropout2d(p=.2),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(16, 32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Dropout2d(p=.2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Dropout2d(p=.2),

        )

        self.fc1 = nn.Sequential(
            nn.Linear(32*50*50, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 30),
            nn.ReLU(inplace=True),

            nn.Linear(30, 2),
            nn.Softmax())

    def forward(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

def test(net):
    dataset = EycDataset()
    net = net.eval()
    dataloader = DataLoader(dataset,
                            shuffle=True,
                            num_workers=8,
                            batch_size=1)

    data_iter = iter(dataloader)

    count = 0

    for i in range(200):
        
        (img, label) = next(data_iter)
        # print(img0)
        
        img = Variable(img).cuda()

        # print(img0)
        (img_output)  = net(img)

        print(img_output, label)

        # time.sleep(0.2)

        # if img_output == label:
        #     count += 1
        
        # print(count)
        # with open("distances.csv", "a") as distancesFile:
        #     distancesFile.write(str(same_distance) + ",0\n" 
        #     + str(diff_distance) + ",1\n")

        # if same_distance > 10:
        #     count_same+=1
        # if diff_distance < 10:
        #     count_diff+=1
        
        # print(count_same, " - ", count_diff)

    print(count)


if __name__ == "__main__":

    epoch_num = 1000
    image_num = 800
    train_batch_size = 64
    iteration_number = 0
    counter = []
    loss_history = []

    dataset = EycDataset(train=True)
    net = ClassifierNetwork().cuda()

    train_dataloader = DataLoader(dataset,
                            shuffle=True,
                            num_workers=8,
                            batch_size=train_batch_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(),lr = 0.0005 )

    for epoch in range(0,epoch_num):
        for i, data in enumerate(train_dataloader):
            (img, label) = data
            img, label = Variable(img).cuda(), Variable(label).cuda()
            (img_output)  = net(img)
            optimizer.zero_grad()
            loss = criterion(img_output, label)
            loss.backward()
            optimizer.step()

            if i%10 == 0:    
                print("Epoch number {}\n Current loss {}\n".format(epoch,loss.data[0]))
                with open("loss_history-patch.csv", 'a') as loss_history:
                    loss_history.write(str(epoch) +
                    ","+str(loss.data[0])+"\n")
            
        if epoch%5 == 0:
            test(net)

        print("Saving model")
        torch.save(net, 'model_patch.pt')
        print("-- Model Checkpoint saved ---")