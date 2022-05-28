from random import random
import torchvision
import torch
from torch.utils.data import random_split
from torchvision import transforms
import numpy as np

base_transform = transforms.Compose([
    # you can add other transformations in this list
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.491, 0.482, 0.446],
                                     std=[0.247, 0.243, 0.261])
])


DATAPATH = ".cifar10"

def generate_unbalanced_data(transform, n_clients, n_classes, alpha):


    if transform is None:
        print("Warning : you have submitted a None transform.")
        transform = base_transform

    dataset = torchvision.datasets.CIFAR10(
            DATAPATH, train=True, download=True, 
            transform=transform,
        )

    