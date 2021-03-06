import torchvision
import torch
from torch.utils.data import random_split
from torchvision import transforms
# functions for getting the data

DATAPATH = '.cifar10'

base_transform = transforms.Compose([
    # you can add other transformations in this list
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.491, 0.482, 0.446],
                                     std=[0.247, 0.243, 0.261])
])

def get_train_validation_data(transform=base_transform, train_proportion = 0.8, BATCH_SIZE=256, shuffle=True, n_worker=2):

    if transform is None:
        transform = base_transform
    
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
        num_workers=n_worker,
        pin_memory=True
        )
    valloader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=int(BATCH_SIZE),
        shuffle=True,
        num_workers=n_worker,
        pin_memory=True)

    return trainloader, valloader


def get_training_data(transform=base_transform, BATCH_SIZE=256,shuffle=True, n_worker=2):
    """
    get DataLoader for training data
    """

    if transform is None:
        transform = base_transform

    trainloader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(
            DATAPATH, train=True, download=True, 
            transform=transform
        ),
        batch_size=BATCH_SIZE, shuffle=shuffle,
            num_workers=n_worker,
            pin_memory=True
    )
    return trainloader

def get_testing_data(transform=base_transform, BATCH_SIZE=256, shuffle=True, n_worker=2):
    """
    get DataLoader for testing data
    """

    if transform is None:
        transform = base_transform

    testloader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(
            DATAPATH, train=False, download=True, 
            transform=transform,
        ),
        batch_size=BATCH_SIZE, shuffle=shuffle,
        num_workers = n_worker
    )

    return testloader







