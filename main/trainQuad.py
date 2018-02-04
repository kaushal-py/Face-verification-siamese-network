import numpy as np
from eycDatasetQuad import EycDataset
import torch
from torch.utils.data import DataLoader,Dataset
from torch import optim
from quadLoss import QuadLoss
from siameseQuad import SiameseNetwork
from torch.autograd import Variable

print("modules loaded")

epoch_num = 1000
image_num = 800
train_batch_size = 32
iteration_number = 0
counter = []
loss_history = []

def main():
    dataset = EycDataset(train=True)
    net = SiameseNetwork().cuda()
    print("model loaded")
    # net = torch.load('models/model_quad_1.pt')

    train_dataloader = DataLoader(dataset,
                            shuffle=True,
                            num_workers=8,
                            batch_size=train_batch_size)
    criterion = QuadLoss()
    optimizer = optim.Adam(net.parameters(),lr = 0.0005 )

    for epoch in range(0,epoch_num):
        for i, data in enumerate(train_dataloader):
            # print(data)
            (anchor, positive, negative, negative2) = data
            anchor, positive, negative, negative2 = Variable(anchor).cuda(), Variable(positive).cuda() , Variable(negative).cuda(), Variable(negative2).cuda()
            (anchor_output, positive_output, negative_output, negative2_output)  = net(anchor, positive, negative, negative2)
            optimizer.zero_grad()
            loss_quad = criterion(anchor_output, positive_output, negative_output, negative2_output)
            loss_quad.backward()
            optimizer.step()
            
            if i%10 == 0:
                print("Epoch number {}\n Current loss {}\n".format(epoch,loss_quad.data[0]))
        
        print("Saving model")
        torch.save(net, 'models/model_quad_3.pt')
        print("-- Model Checkpoint saved ---")

# def batches():
#     dataset = EycDataset(train=True)    
#     train_dataloader = DataLoader(dataset,
#                             shuffle=True,
#                             num_workers=8,
#                             batch_size=train_batch_size)

#     for epoch in range(0,epoch_num):
#         for i, data in enumerate(train_dataloader):
#             print(data.numpy())

main()