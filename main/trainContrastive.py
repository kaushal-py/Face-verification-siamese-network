import numpy as np
from eycDatasetContrastive import EycDataset
import torch
from torch.utils.data import DataLoader,Dataset
from torch import optim
from contrastiveLoss import ContrastiveLoss
from siameseContrastive import SiameseNetwork
from torch.autograd import Variable

epoch_num = 1000
image_num = 800
train_batch_size = 64
iteration_number = 0
counter = []
loss_history = []

dataset = EycDataset(train=True)
net = torch.load("model.pt")

train_dataloader = DataLoader(dataset,
                        shuffle=True,
                        num_workers=8,
                        batch_size=train_batch_size)
criterion = ContrastiveLoss()
optimizer = optim.Adam(net.parameters(),lr = 0.0005 )

for epoch in range(0,epoch_num):
    for i, data in enumerate(train_dataloader):
        (img0, img1, label) = data
        img0, img1, label = Variable(img0).cuda(), Variable(img1).cuda(), Variable(label).cuda()
        (img0_output, img1_output)  = net(img0, img1)
        optimizer.zero_grad()
        loss_contrastive = criterion(img0_output, img1_output, label)
        loss_contrastive.backward()
        optimizer.step()

        if i%10 == 0:    
            print("Epoch number {}\n Current loss {}\n".format(epoch,loss_contrastive.data[0]))
            with open("loss_history-3_6.csv", 'a') as loss_history:
                loss_history.write(str(epoch) +
                ","+str(loss_contrastive.data[0])+"\n")

    print("Saving model")
    torch.save(net, 'model.pt')
    print("-- Model Checkpoint saved ---")