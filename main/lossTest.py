import torch
import torch.nn.functional as F

class Loss(torch.nn.Module):
    """
    Triplet loss function.
    """

    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, distance, label):
        # print(torch.mean(distance).type(torch.cuda.LongTensor))
        loss = torch.mean((label.type(torch.cuda.FloatTensor)) * torch.abs((1.0 - distance)) +
            (1-label.type(torch.cuda.FloatTensor)) * torch.abs(2.0 - distance))
        
        return loss