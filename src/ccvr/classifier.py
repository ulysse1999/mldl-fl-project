import torch, torch.nn as nn

class Classifier(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1 << 11, 10)

    def forward(self, x):
        return self.fc(x)