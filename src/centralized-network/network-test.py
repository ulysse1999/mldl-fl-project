import torch
from resnet50 import ResNet
from torchsummary import summary

resnet = ResNet()
resnet.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
summary(resnet, (3, 224, 224))
