from resnet50 import ResNet
import torch, torch.nn
import torchvision
from torch.optim import SGD
from torch.nn import CrossEntropyLoss


class Client:

    def __init__(self, normalization, local_dataset, batch_size=32 ,epochs=1):
        self.normalization = normalization
        self.dataset = torch.utils.data.DataLoader(
            local_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
            )
        self.epochs=epochs


    