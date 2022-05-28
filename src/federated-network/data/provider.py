from random import random
import torchvision
import torch
from torch.utils.data import random_split
from torchvision import transforms


DATAPATH = '.cifar10'

N_CLIENTS = 100
N_IMAGES_PER_CLIENT = 500

base_transform = transforms.Compose([
    # you can add other transformations in this list
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.491, 0.482, 0.446],
                                     std=[0.247, 0.243, 0.261])
])

def get_dataset(transform=base_transform):

    if transform is None:
        transform = base_transform

    dataset = torchvision.datasets.CIFAR10(
            DATAPATH, train=True, download=True, 
            transform=transform,
        )

    return dataset


def get_iid_split(dataset):

    subdatasets = random_split(dataset,
        [N_IMAGES_PER_CLIENT]*N_CLIENTS)

    return subdatasets


def get_testing_data(transform=base_transform, BATCH_SIZE=1, shuffle=True, n_worker=2):
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