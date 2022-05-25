from data.provider import get_dataset, get_iid_split
from data.augmentation import get_transform


def main():

    transform = get_transform()
    dataset = get_dataset(transform)
    subdatasets = get_iid_split(dataset)



    # split the data

    # create clients and server

    # send data to the clients

    # training loop