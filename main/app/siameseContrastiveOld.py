from torch import nn
import torch

class SiameseNetworkOld(nn.Module):
    def __init__(self):
        super(SiameseNetworkOld, self).__init__()
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

            nn.ReflectionPad2d(1),
            nn.Conv2d(16, 16, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.Dropout2d(p=.2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(16, 32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Dropout2d(p=.2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Dropout2d(p=.2),
        )

        self.fc = nn.Sequential(
            nn.Linear(32*50*50, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            
            nn.Linear(512, 128),
        )

        self.fc_combined = nn.Sequential(
            nn.Linear(128*2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),

            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),

            nn.Linear(32, 2),            
            nn.Softmax()            
        )
        
        
    def forward(self, img0, img1):
        output0 = self.cnn1(img0)
        output0 = output0.view(output0.size()[0], -1)
        output0 = self.fc(output0)

        output1 = self.cnn1(img1)
        output1 = output1.view(output1.size()[0], -1)
        output1 = self.fc(output1)

        output = torch.cat((output0, output1), 1)

        output = self.fc_combined(output)
        return output