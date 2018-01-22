import numpy as np
from eycDatasetContrastive import EycDataset
import torch
from torch.utils.data import DataLoader,Dataset
from torch import optim
from tripletLoss import TripletLoss
from siameseContrastive import SiameseNetwork
from torch.autograd import Variable
import torch.nn.functional as F
from PIL import Image

dataset = EycDataset()
net = torch.load('model.pt').eval()

dataloader = DataLoader(dataset,
                        shuffle=True,
                        num_workers=8,
                        batch_size=1)

data_iter = iter(dataloader)

count = 0

for i in range(200):
    
    (img0, img1, label) = next(data_iter)
    # print(img0)
    
    img0, img1 = Variable(img0).cuda(), Variable(img1).cuda()
    (img0_output, img1_output)  = net(img0, img1)

    euclidean_distance = F.pairwise_distance(img0_output, img1_output)

    euclidean_distance = euclidean_distance.data.cpu().numpy()[0][0]

    label = label.cpu().numpy()[0]
    
    print(euclidean_distance, label)

    if (euclidean_distance < 1 and label == 1) or (euclidean_distance > 1 and label == 0):
        count += 1
    
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
