import numpy as np
from eycDataset import EycDataset
import torch
from torch.utils.data import DataLoader,Dataset
from torch import optim
from tgLoss import TGLoss
from tripletLoss import TripletLoss
from siamese import SiameseNetwork
from torch.autograd import Variable
import test

epoch_num = 30
image_num = 800
train_batch_size = 100
iteration_number = 0
counter = []
loss_history = []

def main():
    dataset = EycDataset(train=True)
    net = SiameseNetwork().cuda()
    # net = torch.load('models/model_triplet_pr_po_max_pool_fix.pt')
    print("model loaded")
    
    train_dataloader = DataLoader(dataset,
                            shuffle=True,
                            num_workers=8,
                            batch_size=train_batch_size)
    # criterion = TGLoss()
    criterion = TripletLoss()
    optimizer = optim.Adam(net.parameters(),lr = 0.0005 )

    for epoch in range(0,epoch_num):
        if epoch%10 == 0:
            test.test(True)
            test.test(False)

        for i, data in enumerate(train_dataloader):
            
            # print(data)
            (anchor, positive, negative) = data
            
            anchor, positive, negative = Variable(anchor).cuda(), Variable(positive).cuda() , Variable(negative).cuda()
            (anchor_output, positive_output, negative_output)  = net(anchor, positive, negative)
            
            optimizer.zero_grad()
            # loss = criterion(anchor_output, positive_output, negative_output, train_batch_size)
            loss = criterion(anchor_output, positive_output, negative_output)
            loss.backward()
            optimizer.step()

            if i%10 == 0:
                print("Epoch number {}\n Current loss {}\n".format(epoch,loss.data[0]))
        

        print("Saving model")
        torch.save(net, 'models/model_triplet_pr_po_max_pool_fix_weighted.pt')
        print("-- Model Checkpoint saved ---")


main()
        
