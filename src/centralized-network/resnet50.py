import torch, torch.nn as nn
from functools import partial

# inspired from pytorch implementation : https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py

# you can check https://colab.research.google.com/github/pytorch/pytorch.github.io/blob/master/assets/hub/nvidia_deeplearningexamples_resnet50.ipynb#scrollTo=modern-finish
# that the architecture is the same (except for the FC layer since we are training on CIFAR10 instead of ImageNet)

# https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
# https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
# https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
# https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html
# https://pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html
# https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
# https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html
# https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool2d.html
# https://pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html

EXPANSION_RATIO=4

class BottleneckBlock(nn.Module):
    # single bottleneck block for large resnets (>= 50 layers)

    def __init__(self, input_channel_s, output_channel_s, downsample, norm):
        super().__init__()

        self.downsample = downsample

        self.conv1 = nn.Conv2d(input_channel_s, output_channel_s, kernel_size=1, stride= 1)
        self.conv2 = nn.Conv2d(output_channel_s, output_channel_s, kernel_size=3, stride=2 if downsample else 1, padding=1)
        self.conv3 = nn.Conv2d(output_channel_s, output_channel_s*EXPANSION_RATIO, kernel_size=1, stride=1)

        self.shortcut = nn.Sequential()

        if downsample or input_channel_s != output_channel_s * EXPANSION_RATIO :
            self.shortcut = nn.Sequential(
                nn.Conv2d(input_channel_s, output_channel_s*EXPANSION_RATIO, kernel_size=1, stride=2 if self.downsample else 1),
                norm(output_channel_s*EXPANSION_RATIO)
            )

        self.n1 = norm(output_channel_s)
        self.n2 = norm(output_channel_s)
        self.n3 = norm(output_channel_s*EXPANSION_RATIO)


    def forward(self, x):

        shortcut = self.shortcut(x)
        x = nn.ReLU()(self.n1(self.conv1(x)))
        x = nn.ReLU()(self.n2(self.conv2(x)))
        x = nn.ReLU()(self.n3(self.conv3(x)))
        x = shortcut + x

        return nn.ReLU()(x)



class ResNet(nn.Module):
    def __init__(self, normalization="batch"):
        super().__init__()

        if normalization=="batch":
            norm = nn.BatchNorm2d
        elif normalization=="group":

            def group_norm(num_channels):
                gn = nn.GroupNorm(num_groups=2, num_channels=num_channels)
                return gn

            norm = group_norm

        self.norm_f = norm
        
        self.input_channel_s = 64

        # CONV, kernel=7, channel=64, stride=2
        # MAXPOOL, kernel=3, stride=2

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.n = norm(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride=2, padding=1)

        # 3 times
        # CONV kernel=1, channels=64
        # CONV kernel=3 channels=64
        # CONV kernel=1 channels=256

        self.conv2x = self._make_layer(3, 64, downsample_first_layer=False)
        
        # DOWNSAMPLING 256 -> 128
        # 4 times :
        # CONV kernel=1, channels=128
        # CONV kernel=3 channels=128
        # CONV kernel=1 channels=512

        self.conv3x = self._make_layer(4, 128, downsample_first_layer=True)
        

        # DOWNSAMPLING 512 -> 256
        # 6 times :
        # CONV kernel=1, channels=256
        # CONV kernel=3 channels=256
        # CONV kernel=1 channels=1024

        self.conv4x = self._make_layer(6, 256, downsample_first_layer=True)
        

        # DOWNSAMPLING 1024 -> 512
        # 3 times :
        # CONV kernel=1, channels=512
        # CONV kernel=3 channels=512
        # CONV kernel=1 channels=2048

        self.conv5x = self._make_layer(3, 512, downsample_first_layer=True)

        # average pool
        # linear FC 1000
        # NO NEED of softmax as it is already included in CrossEntropyLoss which we will use as loss
        # https://stackoverflow.com/questions/64519911/do-i-have-to-add-softmax-in-def-forward-when-i-use-torch-nn-crossentropyloss

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1<<11, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.n(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2x(x)
        x = self.conv3x(x)
        x = self.conv4x(x)
        x = self.conv5x(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)

        return x


    def _make_layer(self, n_blocks, planes, downsample_first_layer=True):

        layers = list()

        layers.append(BottleneckBlock(self.input_channel_s, planes, downsample=downsample_first_layer, norm=self.norm_f))
        self.input_channel_s = planes * EXPANSION_RATIO

        for _ in range(n_blocks-1):
            layers.append(BottleneckBlock(self.input_channel_s, planes, downsample=False, norm=self.norm_f))

        return nn.Sequential(*layers)




        


        
