from data.provider import get_dataset, get_iid_split, N_CLIENTS, N_IMAGES_PER_CLIENT
from data.augmentation import get_transform
from client import Client, ClientSimulation
from server import Server
from argparse import ArgumentParser
from random import sample
from test import test_accuracy
import torch
import gc
from unbalancing.provider import generate_niid_unbalanced_data

N_CLASSES= 10


def main(normalization, epochs, rounds, batch_size, distrib, path):

    
    transform = get_transform()
    dataset = get_dataset(transform)

    # get data and split it
    if distrib=="iid":
        subdatasets = get_iid_split(dataset)
    else:
        subdatasets = generate_niid_unbalanced_data(dataset, N_CLIENTS, N_CLASSES, alpha=0.5)

    sim = ClientSimulation(N_CLIENTS, normalization)

    # create clients

    clients = dict()

    for i in range(N_CLIENTS):
        clients[i] = Client(i, normalization, subdatasets[i]['index'], subdatasets[i]['data'], batch_size, epochs)

    # create server

    server = Server(normalization)

    if path is not None:
        server.update_model(torch.load(path))

    # first round

    gc.collect()

    print(f"##### ROUND 1")
    
    learnings = sim.train(clients)

    pred = {}

    for index in clients:

        pred[index] = server.train(learnings[index])

    server_logit = pred

    # training loop
    
    for round in range(2,rounds+1):

        gc.collect()

        print(f"##### ROUND {round}")

        learnings = sim.train(clients, server_logit)

        pred = {}

        for index in clients:

            pred[index] = server.train(learnings[index])

        server_logit = pred

        if round%20==0:
            server.test_global()


        if round%100==0:
            server.save_model()

    
    




if __name__=='__main__':
    parser = ArgumentParser()

    parser.add_argument("--normalization", type=str, choices=["group", "batch"], required=True, help="Normalization layer, group or batch")
    parser.add_argument("--epochs", type=int, required=True, help="Number of epochs for local training of clients")
    parser.add_argument("--rounds", type=int, required=True, help="Number of rounds of training")
    parser.add_argument("--batchsize", type=int, required=True, help="Batch size during learning")
    parser.add_argument("--path", type=str, required=False, default=None, help="path of a previous model")
    parser.add_argument("--distrib", type=str, required=True, choices=["iid", "niid"])

    args = parser.parse_args()
    main(args.normalization, args.epochs, args.rounds, args.batchsize, args.client_proportion, args.distrib, args.path)


