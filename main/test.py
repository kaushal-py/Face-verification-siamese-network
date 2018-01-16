import numpy as np
from eycDataset import EycDataset
import torch
from torch.utils.data import DataLoader,Dataset
from torch import optim
from tripletLoss import TripletLoss
from siamese import SiameseNetwork
from torch.autograd import Variable
import torch.nn.functional as F

dataset = EycDataset(train=True)
net = torch.load('model1.pt')

dataloader = DataLoader(dataset,
                        shuffle=False,
                        num_workers=8,
                        batch_size=1)

data_iter = iter(dataloader)

count_same = 0
count_diff = 0

for i in range(1000):
    
    anchor, positive, negative = next(data_iter)

    anchor, positive, negative = Variable(anchor).cuda(), Variable(positive).cuda() , Variable(negative).cuda()
    (anchor_output, positive_output, negative_output)  = net(anchor, positive, negative)
    
    same_distance = F.pairwise_distance(anchor_output, positive_output)
    diff_distance = F.pairwise_distance(anchor_output, negative_output)

    same_distance = same_distance.data.cpu().numpy()[0][0]
    diff_distance = diff_distance.data.cpu().numpy()[0][0]
    
    print(i)
    with open("distances.csv", "a") as distancesFile:
        distancesFile.write(str(same_distance) + ",0\n" 
        + str(diff_distance) + ",1\n")

    # if same_distance > 10:
    #     count_same+=1
    # if diff_distance < 10:
    #     count_diff+=1
    
    # print(count_same, " - ", count_diff)
