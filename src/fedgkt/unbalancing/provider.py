from random import choices, choice
import torchvision
import torch
from torch.utils.data import random_split
from torchvision import transforms
import numpy as np
from torch.utils.data import Dataset, Subset


base_transform = transforms.Compose([
    # you can add other transformations in this list
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.491, 0.482, 0.446],
                                     std=[0.247, 0.243, 0.261])
])


DATAPATH = ".cifar10"

def generate_niid_unbalanced_data(dataset, n_clients, n_classes, alpha, batchsize):

    label = np.array(dataset.targets)

    data_by_label = {
        k : np.where(label==k)[0] for k in range(n_classes)
    }

    # distribution of classes for each client
    classes_over_clients = np.random.dirichlet([alpha]*n_classes, n_clients)

    #distribution of data over clients
    data_per_client = np.rint(np.random.dirichlet([5]*n_clients)*50000)

    # for BN to work, you need to avoid cases where AMOUNT_OF_DATA % BATCHSIZE != 0
    for i in range(len(data_per_client)):
        if data_per_client[i] % batchsize != 0 :
            data_per_client[i] -= data_per_client[i] % batchsize
        # lose a bit of data, not a big deal

    result = dict()

    for client in range(n_clients):

        client_dataset = []
        
        class_ids = choices(range(n_classes), weights=classes_over_clients[client], k=int(data_per_client[client]))
        for lab in class_ids:
            i= choice(data_by_label[lab])
            client_dataset.append(i)

        result[client] = Subset(dataset, client_dataset)

    return result
        

        

    