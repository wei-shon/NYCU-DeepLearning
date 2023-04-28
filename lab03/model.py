import numpy as np
import torch
import torch.nn as nn

class EEGNet(nn.Module):
    def __init__(self,activation):
        super(EEGNet, self).__init__()
        self.device = torch.device("cuda")
        # Layer 1
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 51), stride=(1,1) ,padding = (0,25),bias = False),
            nn.BatchNorm2d(16, eps=1e-05 , momentum=0.1 , affine = True , track_running_stats = True)
        )
        # Layer 2
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(2, 1), stride=(1,1) ,groups=16,bias = False),
            nn.BatchNorm2d(32, eps=1e-05 , momentum=0.1 , affine = True , track_running_stats = True),
            activation,
            nn.AvgPool2d(kernel_size=(1,4),stride=(1,4),padding = 0),
            nn.Dropout(p=0.25)
        )
        # Layer 3
        self.separableConv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 15), stride=(1,1) ,padding = (0,7),bias = False),
            nn.BatchNorm2d(32, eps=1e-05 , momentum=0.1 , affine = True , track_running_stats = True),
            activation,
            nn.AvgPool2d(kernel_size=(1,8),stride=(1,8),padding = 0),
            nn.Dropout(p=0.25)
        )
        # fc Layer
        self.classify = nn.Sequential(nn.Linear(736,2 , bias = True))

    def forward(self, x):
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        x = x.view(x.shape[0],-1)
        output = self.classify(x)
        return output
    
class DepConvNet(nn.Module):
    def __init__(self,activation):
        super(DepConvNet, self).__init__()
        self.device = torch.device("cuda")
        self.ficrtconv = nn.Conv2d(1, 25, kernel_size=(1, 5))
        self.convblock1 = nn.Sequential(
            nn.Conv2d(25, 25, kernel_size=(2, 1)),
            nn.BatchNorm2d(25, eps=1e-05 , momentum=0.1),
            activation,
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=0.5)
        )
        self.convblock2 = nn.Sequential(
            nn.Conv2d(25, 50, kernel_size=(1, 5)),
            nn.BatchNorm2d(50, eps=1e-5 , momentum=0.1),
            activation,
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=0.5)
        )
        self.convblock3 = nn.Sequential(
            nn.Conv2d(50, 100, kernel_size=(1, 5)),
            nn.BatchNorm2d(100, eps=1e-5 , momentum=0.1),
            activation,
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=0.5)
        )
        self.convblock4 = nn.Sequential(
            nn.Conv2d(100, 200, kernel_size=(1, 5)),
            nn.BatchNorm2d(200, eps=1e-5 , momentum=0.1),
            activation,
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=0.5)
        )
        self.classify = nn.Sequential(nn.Linear(8600,2))
    
    def forward(self,x):
        x = self.ficrtconv(x)
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = x.view(x.shape[0],-1)
        output = self.classify(x)
        return output
    
