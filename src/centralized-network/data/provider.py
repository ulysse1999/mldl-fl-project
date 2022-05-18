import torchvision
import torch

# functions for getting the data

DATAPATH = '.cifar10'

def get_train_validation_data(transform, train_proportion = 0.8, BATCH_SIZE=256, shuffle=True, n_worker=8):
    
    dataset = torchvision.datasets.CIFAR10(
            DATAPATH, train=True, download=True, 
            transform=transform,
            num_workers=n_worker
        )
    test_abs = int(len(dataset) * train_proportion)

    train_subset, val_subset = random_split(
        dataset, [test_abs, len(trainset) - test_abs])

    trainloader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=8)
    valloader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=8)

    return trainloader, valloader

def get_training_data(transform, BATCH_SIZE=256,shuffle=True, n_worker=8):
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

def get_testing_data(transform, BATCH_SIZE=256, shuffle=True, n_worker=8):
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







