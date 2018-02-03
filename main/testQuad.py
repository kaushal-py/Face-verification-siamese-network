import numpy as np
from eycDatasetQuad import EycDataset
import torch
from torch.utils.data import DataLoader,Dataset
from torch import optim
from quadLoss import QuadLoss
from siameseQuad import SiameseNetwork
from torch.autograd import Variable
import torch.nn.functional as F

dataset = EycDataset()
net = torch.load('models/model_quad_2.pt')

dataloader = DataLoader(dataset,
                        shuffle=False,
                        num_workers=8,
                        batch_size=1)

data_iter = iter(dataloader)

count_same = 0
count_diff = 0

for i in range(200):
    
    anchor, positive, negative, negative2 = next(data_iter)

    anchor, positive, negative, negative2 = Variable(anchor).cuda(), Variable(positive).cuda() , Variable(negative).cuda(), Variable(negative2).cuda()
    (anchor_output, positive_output, negative_output, negative2_output)  = net(anchor, positive, negative, negative2)
    
    same_distance = F.pairwise_distance(anchor_output, positive_output)
    diff_distance = F.pairwise_distance(anchor_output, negative_output)

    same_distance = same_distance.data.cpu().numpy()[0][0]
    diff_distance = diff_distance.data.cpu().numpy()[0][0]
    
    print(same_distance, diff_distance)
    if same_distance > 1:
        count_same+=1
    if diff_distance < 1:
        count_diff+=1
    
print(count_same, " - ", count_diff)