from torch import nn
import torchvision.models as models

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        
        self.pretrained_model = models.resnet18(pretrained=True)

        self.num_ftrs = self.pretrained_model.fc.in_features
        
        self.fc1 = nn.Sequential(
            nn.Linear(self.num_ftrs, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 128))

        self.pretrained_model.fc = self.fc1

    def forward_once(self, x):
        output = self.pretrained_model(x)
        # output = output.view(output.size()[0], -1)
        # output = self.fc1(output)
        return output

    def forward(self, img0, img1):
        img0_output = self.forward_once(img0)
        img1_output = self.forward_once(img1)
        return img0_output, img1_output