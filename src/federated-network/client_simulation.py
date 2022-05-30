from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from resnet50 import ResNet
import torch
from joblib import Parallel, delayed

class ClientSimulation:

    def __init__(self, n_clients, normalization):
        self.n_clients = n_clients
        self.normalization = normalization
        

    def train(self, clients, client_subset, server_model_dict):

        self.cl_data = dict()

        Parallel(n_jobs=2, require="sharedmem")(delayed(self._train(clients, client_index, server_model_dict)) for client_index in client_subset)
        
        return self.cl_data

    def _train(self, clients, client_index, server_model_dict):
        cl = _Client(self.normalization, clients[client_index].dataset, clients[client_index].epochs, server_model_dict)
        cl.train()
        self.cl_data[client_index] = cl

class _Client:

    def __init__(self, normalization, local_dataset, epochs, model_dict):
        self.model = ResNet(normalization)
        self.model.load_state_dict(model_dict)
        self.dataset = local_dataset
        self.epochs = epochs

    def train(self):

        optimizer = SGD(self.model.parameters(), lr=1e-3, weight_decay=5e-4)

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

