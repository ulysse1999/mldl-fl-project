import torch
from resnet50 import ResNet
from torchinfo import summary

resnet = ResNet()
resnet.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
batch_size = 1 # testing for now
summary(resnet, (batch_size, 3, 224, 224))

