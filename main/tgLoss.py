import torch
import torch.nn.functional as F

class TGLoss(torch.nn.Module):
    """
    Triplet loss function.
    """

    def __init__(self, margin=2.0, l=2.0):
        super(TGLoss, self).__init__()
        self.margin = margin
        self.l = l

    def forward(self, anchor, positive, negative, N):
        positive_distance = F.pairwise_distance(anchor, positive)
        negative_distance = F.pairwise_distance(anchor, negative)

        distance_dist_positive = torch.pow(positive_distance, 2)/4
        distance_dist_negative = torch.pow(negative_distance, 2)/4
    
        mean_positive = torch.sum(distance_dist_positive)/N
        mean_negative = torch.sum(distance_dist_negative)/N

        var_positive = torch.sum(torch.pow((distance_dist_positive - mean_positive), 2))/N
        var_negative = torch.sum(torch.pow((distance_dist_negative - mean_negative), 2))/N

        loss = ((var_positive + var_negative) 
                    + self.l*torch.clamp(self.margin + mean_positive -  mean_negative, 
                        min=0.0))

        return loss