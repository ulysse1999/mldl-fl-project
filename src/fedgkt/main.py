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


def main(normalization, epochs, rounds, batch_size, distrib, path, alpha):

    
    transform = get_transform()
    dataset = get_dataset(transform)

    # get data and split it
    if distrib=="iid":
        subdatasets = get_iid_split(dataset)
    else:
        subdatasets = generate_niid_unbalanced_data(dataset, N_CLIENTS, N_CLASSES, alpha=alpha, batchsize=batch_size)

    sim = ClientSimulation(N_CLIENTS, normalization)

    # create clients

    clients = dict()

    for i in range(1):
        clients[i] = Client(i, normalization, subdatasets[i], batch_size, epochs)

    # create server

    server = Server(normalization, epochs)

    if path is not None:
        server.update_model(torch.load(path))

    # first round

    gc.collect()

    print(f"##### ROUND 1 #####\n")
    
    learnings = sim.train(clients)

    pred = {}

    print(f"Training server")

    for index in clients:
        pred[index] = server.train(learnings[index])

    print(f"Done\n")

    server_logit = pred

    test_accuracy(clients[0].model, server.model)

    # training loop
    
    for round in range(2,rounds+1):

        gc.collect()

        print(f"##### ROUND {round} #####")

        learnings = sim.train(clients, server_logit)

        pred = {}

        for index in clients:

            pred[index] = server.train(learnings[index])

        server_logit = pred

        test_accuracy(clients[0].model, server.model)


        if round%5==0:
            server.save_model()

    
    




if __name__=='__main__':
    parser = ArgumentParser()

    parser.add_argument("--normalization", type=str, choices=["group", "batch"], required=True, help="Normalization layer, group or batch")
    parser.add_argument("--epochs", type=int, required=True, help="Number of epochs for local training of clients")
    parser.add_argument("--rounds", type=int, required=True, help="Number of rounds of training")
    parser.add_argument("--batchsize", type=int, required=True, help="Batch size during learning")
    parser.add_argument("--path", type=str, required=False, default=None, help="path of a previous model")
    parser.add_argument("--distrib", type=str, required=True, choices=["iid", "niid"])
    parser.add_argument("--alpha", type=float, required=False, default=1.0, help="Concentration parameter for Dirichlet distribution")

    args = parser.parse_args()
    main(args.normalization, args.epochs, args.rounds, args.batchsize, args.distrib, args.path, args.alpha)



