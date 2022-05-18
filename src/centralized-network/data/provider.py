import torchvision
import torch
from torch.utils.data import random_split

# functions for getting the data

DATAPATH = '.cifar10'

def get_train_validation_data(transform=None, train_proportion = 0.8, BATCH_SIZE=256, shuffle=True, n_worker=4):
    
    dataset = torchvision.datasets.CIFAR10(
            DATAPATH, train=True, download=True, 
            transform=transform,
        )
    train_abs = int(len(dataset) * train_proportion)

    train_subset, val_subset = random_split(
        dataset, [train_abs, len(dataset) - train_abs])

    trainloader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=int(BATCH_SIZE),
        shuffle=True,
        num_workers=n_worker)
    valloader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=int(BATCH_SIZE),
        shuffle=True,
        num_workers=n_worker)

    return trainloader, valloader


def get_training_data(transform=None, BATCH_SIZE=256,shuffle=True, n_worker=4):
    """
    get DataLoader for training data
    """
    trainloader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(
            DATAPATH, train=True, download=True, 
            transform=transform,
            num_workers=n_worker
        ),
        batch_size=BATCH_SIZE, shuffle=shuffle
    )
    return trainloader

def get_testing_data(transform=None, BATCH_SIZE=256, shuffle=True, n_worker=2):
    """
    get DataLoader for testing data
    """

    testloader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(
            DATAPATH, train=False, download=True, 
        ),
        batch_size=BATCH_SIZE, shuffle=shuffle,
        transform=transform,
        num_workers = n_worker
    )

    return testloader







