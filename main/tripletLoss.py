import torch
import torch.nn.functional as F

class TripletLoss(torch.nn.Module):
    """
    Triplet loss function.
    """

    def __init__(self, margin=2.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        positive_distance = F.pairwise_distance(anchor, positive)
        negative_distance = F.pairwise_distance(anchor, negative)

        dist_hinge = torch.clamp(self.margin + positive_distance - negative_distance, min=0.0)
        triplet_loss = torch.mean(dist_hinge)
        
        return triplet_loss
