import torch, torch.nn as nn

class BottleneckBlock(nn.Module):
    # single bottleneck block for large resnets (>= 50 layers)

    def __init__(self, input_channel_s, output_channel_s, stride, normalization):
        super().__init__()


    def forward(self,):

class ResNet(nn.Module):
    def __init__(self,):

        # CONV, kernel=7, channel=64, stride=2
        # MAXPOOL, kernel=3, stride=2

        # 3 times
        # CONV kernel=1, channels=64
        # CONV kernel=3 channels=64
        # CONV kernel=1 channels=256

        # DOWNSAMPLING 256 -> 128

        # 4 times
        # CONV kernel=1, channels=128
        # CONV kernel=3 channels=128
        # CONV kernel=1 channels=512

        # DOWNSAMPLING 512 -> 256

        # 6 times
        # CONV kernel=1, channels=256
        # CONV kernel=3 channels=256
        # CONV kernel=1 channels=1024

        # DOWNSAMPLING 1024 -> 512

        # 3 times
        # CONV kernel=1, channels=512
        # CONV kernel=3 channels=512
        # CONV kernel=1 channels=2048

        # average pool
        # linear FC 1000
        # NO NEED of softmax as it is already included in CrossEntropyLoss which we will use as loss
        



    def forward(self,):