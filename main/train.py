import numpy as np
from eycDataset import EycDataset
from torch.utils.data import DataLoader,Dataset
from torch import optim
from tripletLoss import TripletLoss
from siamese import SiameseNetwork


epoch_num = 100
image_num = 800
train_batch_size = 64
iteration_number = 0
counter = []
loss_history = []

dataset = EycDataset(train=True)
net = SiameseNetwork()

train_dataloader = DataLoader(dataset,
                        shuffle=True,
                        num_workers=8,
                        batch_size=train_batch_size)
criterion = TripletLoss()
optimizer = optim.Adam(net.parameters(),lr = 0.0005 )
print("njdf")
for epoch in range(0,epoch_num):
    for i, data in enumerate(train_dataloader,0):
        (anchor, positive, negative) = data
        # img0, img1 , label = Variable(img0).cuda(), Variable(img1).cuda() , Variable(label).cuda()
        (anchor_output, positive_output, negative_output)  = net(anchor, positive, negative)
        optimizer.zero_grad()
        loss_triplet = criterion(anchor_output, positive_output, negative_output)
        loss_triplet.backward()
        optimizer.step()
        if i %10 == 0 :
            print("Epoch number {}\n Current loss {}\n".format(epoch,loss_triplet.data[0]))
            iteration_number +=10
            counter.append(iteration_number)
            loss_history.append(loss_triplet.data[0])
            np.save("loss_history.npy"+(i%10),loss_history)
