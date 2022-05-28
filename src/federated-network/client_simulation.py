from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from resnet50 import ResNet
import torch

class ClientSimulation:

    def __init__(self, n_clients, normalization):
        self.n_clients = n_clients
        self.normalization = normalization
        

    def train(self, clients, client_subset, server_model_dict):

        cl_data = dict()
        
        for index in client_subset:
            
            cl = _Client(self.normalization, clients[index].dataset, clients[index].epochs, server_model_dict)
            print(f"Training client {index}")
            cl.train()
            print("Done")
            cl_data[index] = cl

        return cl_data


class _Client:

    def __init__(self, normalization, local_dataset, epochs, model_dict):
        self.model = ResNet(normalization)
        self.model.load_state_dict(model_dict)
        self.dataset = local_dataset
        self.epochs = epochs

    def train(self):

        optimizer = SGD(self.model.parameters(), lr=1e-3, weight_decay=0.05)

        criterion = CrossEntropyLoss()
        criterion.cuda()

        self.model.cuda()
        self.model.train()

        for epoch in range(self.epochs):
            # training loop
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

