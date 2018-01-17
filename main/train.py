import numpy as np
from eycDataset import EycDataset
import torch
from torch.utils.data import DataLoader,Dataset
from torch import optim
from tripletLoss import TripletLoss
from siamese_partial import SiameseNetwork
from torch.autograd import Variable

epoch_num = 1000
image_num = 800
train_batch_size = 64
iteration_number = 0
counter = []
loss_history = []

dataset = EycDataset(train=True)
net = SiameseNetwork().cuda()

train_dataloader = DataLoader(dataset,
                        shuffle=True,
                        num_workers=8,
                        batch_size=train_batch_size)
criterion = TripletLoss()
optimizer = optim.Adam(net.parameters(),lr = 0.0005 )

for epoch in range(0,epoch_num):
    for i, data in enumerate(train_dataloader):
        (anchor, positive, negative) = data
        anchor, positive, negative = Variable(anchor).cuda(), Variable(positive).cuda() , Variable(negative).cuda()
        (anchor_output, positive_output, negative_output)  = net(anchor, positive, negative)
        optimizer.zero_grad()
        loss_triplet = criterion(anchor_output, positive_output, negative_output)
        loss_triplet.backward()
        optimizer.step()

        if i%10 == 0:    
            print("Epoch number {}\n Current loss {}\n".format(epoch,loss_triplet.data[0]))
    
    print("Saving model")
    torch.save(net, 'model.pt')
    print("-- Model Checkpoint saved ---")