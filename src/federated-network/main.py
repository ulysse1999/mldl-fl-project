from data.provider import get_dataset, get_iid_split, N_CLIENTS, N_IMAGES_PER_CLIENT
from data.augmentation import get_transform
from client import Client
from server import Server
from argparse import ArgumentParser
from resnet50 import ResNet
from random import sample
from train import test_accuracy


# global parameters : number of epochs locally, normalization type

def average(clients, normalization, client_subset):
    print("Server update")

    n_selected_clients = len(client_subset)

    dummy_model = ResNet(normalization)
    dummy_dict = dummy_model.state_dict()
    for key in dummy_dict:
        print(key)
        dummy_dict[key] = sum([clients[i].get_data(key) for i in client_subset]) / n_selected_clients

    return dummy_dict


def main(epochs, normalization, rounds, client_proportion, batch_size):

    # get data and split it
    transform = get_transform()
    dataset = get_dataset(transform)
    subdatasets = get_iid_split(dataset)

    # create clients

    clients = dict()

    for i in range(N_CLIENTS):
        clients[i] = Client(normalization, subdatasets[i], batch_size,epochs)

    # create server

    server = Server(normalization)


    # training loop

    for _ in range(rounds):

        client_subset = sample(range(N_CLIENTS), int(client_proportion*N_CLIENTS))

        for index in client_subset:
            print(f"Training client  {index}")
            clients[index].set_model(server.model.state_dict())
            clients[index].train()
            print("Done")

        model_dict = average(clients, normalization, client_subset)

        server.update_model(model_dict)

    test_acc = test_accuracy(server.model)
    print("Best trial test set accuracy: {}".format(test_acc))
    




if __name__=='__main__':
    parser = ArgumentParser()

    parser.add_argument("--normalization", type=str, choices=["group", "batch"], required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--batchsize", type=int, required=True)
    parser.add_argument("--client_proportion", type=float, required=True)

    args = parser.parse_args()
    main(args.epochs, args.normalization, args.rounds, args.client_proportion, args.batchsize)



