from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from resnet8 import ResNet8
import torch
from torch.utils.data import Dataset, Subset

class ClientSimulation:

    def __init__(self, n_clients, normalization):
        self.n_clients = n_clients
        self.normalization = normalization


    def train(self, clients, client_subset, server_model_dict):

        cl_data = dict()

        for index in client_subset:

            cl = _Client(self.normalization, clients[index].dataset_index, clients[index].dataset, clients[index].epochs, server_model_dict)
            print(f"Training client {index}")
            cl.train()
            print("Done")
            cl_data[index] = cl

        return cl_data


class _Client:

    def __init__(self, normalization, local_dataset_index, local_dataset, epochs):
        self.model = ResNet8(normalization)
        self.data_index = local_dataset_index
        self.dataset = local_dataset
        self.epochs = epochs
    
    def train(self, server_logit=None):

        optimizer = SGD(self.modelo.parameters(), lr=1e-3, weight_decay=5e-4)
        criterion = CrossEntropyLoss()
        criterion.cuda()

        self.model.cuda()
        self.model.train()

        if server_logit is not None:
            self.dataset = zip(self.dataset[0], Subset(server_logit, self.data_index)) 

        for epoch in range(self.epochs):
            for i, data in enumerate(self.dataset):
                imgs, labels = data 
                imgs, labels = imgs.cuda(), labels.cuda()

                optimizer.zero_grad()
                pred = self.model(imgs)
                pred = pred.cuda()

                loss = criterion(pred, labels)
                loss.backward()
                optimizer.step()

        self.model = self.model.to('cpu')

        self.model_dict = self.model.state_dict()
        torch.cuda.empty_cache()

    def get_data(self, key):
        return self.model_dict[key]


class Client:

    def __init__(self, normalization, local_dataset_index, local_dataset, batch_size=32 ,epochs=1):
        self.normalization = normalization
        self.data_index = local_dataset_index
        self.dataset = torch.utils.data.DataLoader(
            local_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
            )
        self.epochs=epochs