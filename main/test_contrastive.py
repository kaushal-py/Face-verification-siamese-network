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
import time

dataset = EycDataset()
net = torch.load('model_pre_post.pt').eval()

dataloader = DataLoader(dataset,
                        shuffle=False,
                        num_workers=8,
                        batch_size=1)

sum = 0
for k in range(2):
    data_iter = iter(dataloader)
    count = 0
    # count_same = 0
    # (img0, _, _) = next(data_iter)
    # img0 = Variable(img0).cuda()
        
    for i in range(800):
        
        (img0, img1, label) = next(data_iter)
        # print(img0)
        
        img0, img1 = Variable(img0).cuda(), Variable(img1).cuda()
        # img1 = Variable(img1).cuda()

        # print(img0)
        img0_output, img1_output  = net(img0, img1)
        # output = output.data.cpu().numpy()[0]

        # if (output[0] > 0.5 and label[0] == 0) or (output[1] > 0.5 and label[0] == 1):
        #     count += 1
        # print(output.data.cpu().numpy()[0], label[0])

        euclidean_distance = F.pairwise_distance(img0_output, img1_output)

        euclidean_distance = euclidean_distance.data.cpu().numpy()[0][0]

        label = label.cpu().numpy()[0]
        
        print(euclidean_distance, label)

        # time.sleep(0.2)

        if (euclidean_distance < 0.5 and label == 1) or (euclidean_distance > 0.5 and label == 0):
            count += 1
            # print(count)
        
        # if (euclidean_distance < 0.5)
        # with open("distances.csv", "a") as distancesFile:
        #     distancesFile.write(str(same_distance) + ",0\n" 
        #     + str(diff_distance) + ",1\n")

        # if same_distance > 10:
        #     count_same+=1
        # if diff_distance < 10:
        #     count_diff+=1
        
        # print(count_same, " - ", count_diff)

    print(count)
    sum += count

print("Accuracy :", (1600-sum)/16, "%")
