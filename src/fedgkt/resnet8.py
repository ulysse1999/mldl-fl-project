import torch, torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, input_channels, output_channels, norm):
        super().__init__()

        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1)

        self.n1 = norm(output_channels)
        self.n2 = norm(output_channels)
    
    def forward(self, x):

        shortcut = x
        x = nn.ReLU()(self.n1(self.conv1(x)))
        x = nn.ReLU()(self.n2(self.conv2(x)))
        x += shortcut
        x = nn.ReLU()(x)

        return x

        

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

        self.nomr_f = norm

        self.input_channels = 16

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.n = norm(16)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1)

        self.layer1 = self._make_layer(2, 16)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(160, 10)


    def _make_layer(self, n_blocks, planes):

        layers = list()

        for _ in range(n_blocks):
            layers.append(BasicBlock(self.input_channels, planes, self.nomr_f))

        return nn.Sequential(*layers)
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.n(x)
        x = self.relu(x)
        xf = self.layer1(x)
        x = self.avgpool(xf)
        x = x.view(1, -1)
        x = self.fc(x)

        return x, xf

    
    