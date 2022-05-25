from data.provider import get_dataset, get_iid_split, N_CLIENTS, N_IMAGES_PER_CLIENT
from data.augmentation import get_transform
from client import Client
from server import Server
from argparse import ArgumentParser


# global parameters : number of epochs locally, normalization type



def main(epochs, normalization, rounds):

    # get data and split it
    transform = get_transform()
    dataset = get_dataset(transform)
    subdatasets = get_iid_split(dataset)

    # create clients

    clients = dict()

    for i in range(N_CLIENTS):
        clients[i] = Client(subdatasets[i])

    # create server

    server = Server(normalization)


    # training loop

    for _ in range(rounds):
        ...


if __name__=='__main__':
    parser = ArgumentParser()

    parser.add_argument("--normalization", type=str, choices=["group", "batch"], required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--batchsize", type=int, required=True)

    args = parser.parse_args()
    main(args.epochs, args.normalization, args.rounds)



