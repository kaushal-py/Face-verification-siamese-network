import numpy as np
from eycDataset import EycDataset
import torch
from torch.utils.data import DataLoader,Dataset
from torch import optim
from tripletLoss import TripletLoss
from siamese import SiameseNetwork
from torch.autograd import Variable
import torch.nn.functional as F
from torchviz import make_dot, make_dot_from_trace

dataset = EycDataset(train=True)
net = torch.load('models/model_triplet_8.pt').eval()

print(net.layers)
# all_weights = []
# for layer in net.layers:
#    w = layer.get_weights()
#    all_weights.append(w)

all_weights = np.array(all_weights)
np.save('all_weights.npy', all_weights)

# from model import Net
# from trainer import Trainer 
import torch
from torch import nn
from matplotlib import pyplot as plt

model = Net()
ckpt = torch.load('path_to_checkpoint')
model.load_state_dict(ckpt['state_dict'])
filter = model.conv1.weight.data.numpy()
#(1/(2*(maximum negative value)))*filter+0.5 === you need to normalize the filter before plotting.
filter = (1/(2*3.69201088))*filter + 0.5 #Normalizing the values to [0,1]

#num_cols= choose the grid size you want
def plot_kernels(tensor, num_cols=8):
    if not tensor.ndim==4:
        raise Exception("assumes a 4D tensor")
    if not tensor.shape[-1]==3:
        raise Exception("last dim needs to be 3 to plot")
    num_kernels = tensor.shape[0]
    num_rows = 1+ num_kernels // num_cols
    fig = plt.figure(figsize=(num_cols,num_rows))
    for i in range(tensor.shape[0]):
        ax1 = fig.add_subplot(num_rows,num_cols,i+1)
        ax1.imshow(tensor[i])
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()
    
    
plot_kernels(filter)


# dataloader = DataLoader(dataset,
#                         shuffle=False,
#                         num_workers=8,
#                         batch_size=1)

# data_iter = iter(dataloader)

# count_same = 0
# count_diff = 0

# # for i in range(400):
    
# anchor, positive, negative = next(data_iter)

# anchor, positive, negative = Variable(anchor).cuda(), Variable(positive).cuda() , Variable(negative).cuda()
# (anchor_output, positive_output, negative_output)  = net(anchor, positive, negative)

# make_dot( anchor_output, params=dict(list(net.named_parameters()) + [('x', anchor)]))
# print(count_same, " - ", count_diff)
