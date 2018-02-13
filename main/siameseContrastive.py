from torch import nn
import torch
import torch.nn.functional as F

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 4, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),
            nn.Dropout2d(p=.5),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            nn.Dropout2d(p=.5),

            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.Dropout2d(p=.5),

            nn.ReflectionPad2d(1),
            nn.Conv2d(16, 32, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Dropout2d(p=.5),
        )

        self.cnn2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 4, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),
            nn.Dropout2d(p=.5),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            nn.Dropout2d(p=.5),

            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.Dropout2d(p=.5),

            nn.ReflectionPad2d(1),
            nn.Conv2d(16, 32, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Dropout2d(p=.5),
        )

        self.fc = nn.Sequential(
            nn.Linear(32*50*50, 512),
            nn.ReLU(inplace=True),

            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            
            nn.Linear(256, 128)
        )
        
        
    def forward(self, img0, img1):
        output0 = self.cnn1(img0)
        output0 = output0.view(output0.size()[0], -1)
        output0 = self.fc(output0)
        output0 = F.normalize(output0)

        output1 = self.cnn2(img1)
        output1 = output1.view(output1.size()[0], -1)
        output1 = self.fc(output1)
        output1 = F.normalize(output1)
    
        return output0, output1