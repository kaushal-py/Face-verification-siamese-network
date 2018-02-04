import torch
import torch.nn
import torch.nn.functional as F
import math

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        d = F.pairwise_distance(output1, output2)
        p = ((1 + math.exp(-self.margin))
        /(1 + torch.exp(d-self.margin)))

        loss = -torch.mean(label*torch.log(p) + (1-label)*torch.log(1-p))
        return loss