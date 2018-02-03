import torch
import torch.nn.functional as F

class QuadLoss(torch.nn.Module):
    """
    Triplet loss function.
    """

    def __init__(self, alpha1=1.0, alpha2=0.2):
        super(QuadLoss, self).__init__()
        self.alpha1 = alpha1
        self.alpha2 = alpha2

    def forward(self, anchor, positive, negative, negative2):
        positive_distance = F.pairwise_distance(anchor, positive)
        negative_distance = F.pairwise_distance(anchor, negative)
        negative2_distance = F.pairwise_distance(negative, negative2)

        dist_trip = torch.clamp(self.alpha1 + positive_distance - negative_distance, min=0.0)
        dist_quad = torch.clamp(self.alpha2 + positive_distance - negative2_distance, min=0.0)
        quad_loss = torch.mean(dist_trip + dist_quad)
        
        return quad_loss