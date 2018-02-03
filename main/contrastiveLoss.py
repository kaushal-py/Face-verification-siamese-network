import torch
import torch.nn
import torch.nn.functional as F

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        # print(output1)
        loss_contrastive = torch.mean((1-label) * 0.5 * torch.pow(euclidean_distance, 2) +
                                      (label) * 0.5 * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        # loss_contrastive = loss_contrastive
        return loss_contrastive
