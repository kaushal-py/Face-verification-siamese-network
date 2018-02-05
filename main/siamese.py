from torch import nn
import torch.nn.functional as F

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),
            nn.Dropout2d(p=.2),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            nn.Dropout2d(p=.2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 16, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.Dropout2d(p=.2),
        )

        self.cnn2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),
            nn.Dropout2d(p=.2),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            nn.Dropout2d(p=.2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 16, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.Dropout2d(p=.2),
        )

        self.cnn3 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),
            nn.Dropout2d(p=.2),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            nn.Dropout2d(p=.2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 16, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.Dropout2d(p=.2),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(16*50*50, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),

            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),

            nn.Linear(512, 128))

    def forward(self, anchor_input, pos_input, neg_input):
        
        anchor_output = self.cnn1(anchor_input)
        # anchor_output = F.normalize(anchor_output)
        anchor_output = anchor_output.view(anchor_output.size()[0], -1)
        anchor_output = self.fc1(anchor_output)
        anchor_output = F.normalize(anchor_output)

        pos_output = self.cnn1(pos_input)
        # anchor_output = F.normalize(anchor_output)
        pos_output = pos_output.view(pos_output.size()[0], -1)
        pos_output = self.fc1(pos_output)
        pos_output = F.normalize(pos_output)

        neg_output = self.cnn1(neg_input)
        # anchor_output = F.normalize(anchor_output)
        neg_output = neg_output.view(neg_output.size()[0], -1)
        neg_output = self.fc1(neg_output)
        neg_output = F.normalize(neg_output)

        return anchor_output, pos_output, neg_output