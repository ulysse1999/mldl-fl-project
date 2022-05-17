import torchvision
import torch


def get_training_data(BATCH_SIZE=256,shuffle=True):
    trainloader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(
            '.cifar10', train=True, download=True, 
        ),
        batch_size=BATCH_SIZE, shuffle=shuffle
    )
    return trainloader

def get_testing_data(BATCH_SIZE=256, shuffle=True):

    testloader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(
            '.cifar10', train=False, download=True, 
        ),
        batch_size=BATCH_SIZE, shuffle=shuffle
    )

    return testloader







