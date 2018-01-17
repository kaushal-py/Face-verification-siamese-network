import numpy as np
from eycDatasetTest import EycDataset
import torch
from torch.utils.data import DataLoader,Dataset
from torch import optim
from lossTest import Loss
from siameseTest import SiameseNetwork
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
criterion = Loss()
optimizer = optim.Adam(net.parameters(), lr=0.0005)

for epoch in range(0, epoch_num):
    for i, data in enumerate(train_dataloader):
        (img_0, img_1, label) = data
        img_0, img_1, label = Variable(img_0).cuda(), Variable(img_1).cuda() , Variable(label).cuda()
        difference  = net(img_0, img_1)
        optimizer.zero_grad()
        loss_triplet = criterion(difference, label)
        loss_triplet.backward()
        optimizer.step()
        if i %10 == 0 :
            print("Epoch number {}\n Current loss {}\n".format(epoch,loss_triplet.data[0]))
            iteration_number +=10
            counter.append(iteration_number)
            loss_history.append(loss_triplet.data[0])
            np.save("loss_history.npy",loss_history)
    
    torch.save(net, 'model-test.pt')
    print("-- Model Checkpoint saved ---")