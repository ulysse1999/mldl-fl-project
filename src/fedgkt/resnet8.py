import torch, torch.nn as nn


class ResNet8(nn.Module):
    def __init__(self, normalization="batch"):

        super().__init__()

        assert normalization in {"batch", "group"}, f"Normalization should be batch or group, is :{normalization}"

        if normalization=="batch":
            norm = nn.BatchNorm2d
        elif normalization=="group":

            def group_norm(num_channels):
                gn = nn.GroupNorm(num_groups=2, num_channels=num_channels)
                return gn

            norm = group_norm

        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(3, 16, (3,3), stride=(1,1), padding=(1,1))
        self.n1 = norm(16)
        
        self.maxpool = nn.MaxPool2d((3,1))

        self.lay1conv1 =nn.Conv2d(16,16,(3,3),stride=(1,1), padding=(1,1))
        self.lay1n1 = norm(16)

        self.lay1conv2 = nn.Conv2d(16,16,(3,3),stride=(1,1), padding=(1,1))
        self.lay1n2 = norm(16)

        self.lay2conv1 = nn.Conv2d(16,16,(3,3),stride=(1,1), padding=(1,1))
        self.lay2n1 = norm(16)

        self.lay2conv2 = nn.Conv2d(16,16,(3,3),stride=(1,1), padding=(1,1))
        self.lay2n2 = norm(16)


        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        x = self.relu(self.n1(self.conv1(x)))
        # feature extraction should take the value here, that is in "relu" layer
        x = self.maxpool(x)
        print(f"after max pool size: {x.size()}")
        x = nn.ReLU()(self.lay1n1(self.lay1conv1(x)))
        x = nn.ReLU()(self.lay1n2(self.lay1conv2(x)))
        print(f"after first layer size: {x.size()}")
        x = nn.ReLU()(self.lay2n1(self.lay2conv1(x)))
        x = nn.ReLU()(self.lay2n2(self.lay2conv2(x)))
        feats = x
        print(f"features size: {x.size()}")
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        print(f"after avg pool size: {x.size()}")
        x = self.fc(x)

        return x, feats
