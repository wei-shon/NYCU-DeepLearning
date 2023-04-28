import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(Block, self).__init__()
        self.expansion = 4

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x
    
class ResNet50(nn.Module):
    def __init__(self, image_channels = 3 , num_classes = 5  , pretrained = False):
        super(ResNet50, self).__init__()
        self.expansion = 4
        layers = [3, 4, 6, 3]
        self.in_channels = 64

        self.pretrained = pretrained
        if  pretrained == True:
            model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
            self.conv1 = getattr(model , 'conv1')
            self.bn1 = getattr(model , 'bn1')
            self.relu = getattr(model , 'relu')
            self.maxpool = getattr(model , 'maxpool')
            self.layer1 = getattr(model , 'layer1')
            self.layer2 = getattr(model , 'layer2')
            self.layer3 = getattr(model , 'layer3')
            self.layer4 = getattr(model , 'layer4')
            self.avgpool = getattr(model , 'avgpool')
        else:
            self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU()
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            # ResNetLayers
            self.layer1 = self.__make_layer(layers[0], intermediate_channels=64, stride=1)
            self.layer2 = self.__make_layer(layers[1], intermediate_channels=128, stride=2)
            self.layer3 = self.__make_layer(layers[2], intermediate_channels=256, stride=2)
            self.layer4 = self.__make_layer(layers[3], intermediate_channels=512, stride=2)

            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * self.expansion, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

    def __make_layer(self, num_residual_blocks, intermediate_channels, stride):
        layers = []

        identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, intermediate_channels*self.expansion, kernel_size=1, stride=stride),
                                            nn.BatchNorm2d(intermediate_channels*self.expansion))
        layers.append(Block( self.in_channels, intermediate_channels, identity_downsample, stride))
        self.in_channels = intermediate_channels * self.expansion # 256
        for i in range(num_residual_blocks - 1):
            layers.append(Block( self.in_channels, intermediate_channels)) # 256 -> 64, 64*4 (256) again
        return nn.Sequential(*layers)
