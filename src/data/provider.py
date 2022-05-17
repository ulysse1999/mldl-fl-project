import torchvision
import torch

# functions for getting the data

def get_training_data(transform, BATCH_SIZE=256,shuffle=True, n_worker=8):
    """
    get DataLoader for training data
    """
    trainloader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(
            '.cifar10', train=True, download=True, 
            transform=transform,
            num_workers=n_worker
        ),
        batch_size=BATCH_SIZE, shuffle=shuffle
    )
    return trainloader

def get_testing_data(transform, BATCH_SIZE=256, shuffle=True, n_worker=8):
    """
    get DataLoader for testing data
    """

    testloader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(
            '.cifar10', train=False, download=True, 
        ),
        batch_size=BATCH_SIZE, shuffle=shuffle,
        transform=transform,
        num_workers = n_worker
    )

    return testloader







