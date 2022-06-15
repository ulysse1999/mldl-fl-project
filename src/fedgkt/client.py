from torch.optim import SGD
from torch.nn import CrossEntropyLoss, KLDivLoss
from resnet8 import ResNet8
import torch
from torch.utils.data import Dataset, Subset, TensorDataset
import numpy as np

class ClientSimulation:

    def __init__(self, n_clients, normalization):
        self.n_clients = n_clients
        self.normalization = normalization


    def train(self, clients, server_logit=None):

        learnings = dict()

        for index in clients:

            print(f"Training client {index}")
            cl_learnings = clients[index].train(server_logit)
            print("Done")
            learnings[index] = cl_learnings

        return learnings


class Client:

    def __init__(self, index, normalization, local_dataset_index, local_dataset, batch_size=32, epochs=1):
        self.index = index
        self.model = ResNet8(normalization)
        self.data_index = local_dataset_index
        self.dataset = torch.utils.data.DataLoader(
            local_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
            )
        self.epochs = epochs
    
    def train(self, server_logit=None):

        optimizer = SGD(self.model.parameters(), lr=1e-3, weight_decay=5e-4)
        crossEntropy = CrossEntropyLoss()
        KLDiv = KLDivLoss()
        crossEntropy.cuda()
        KLDiv.cuda()

        self.model.cuda()
        self.model.train()

        if server_logit is not None:
            self.dataset = TensorDataset(self.dataset[0,:], server_logit[self.index]) 

        for epoch in range(self.epochs-1):
            for i, data in enumerate(self.dataset):
                imgs, labels = data 
                imgs, labels = imgs.cuda(), labels.cuda()

                optimizer.zero_grad()
                pred, feats = self.model(imgs)
                pred = pred.cuda()

                loss = crossEntropy(pred, labels) # +KLDiv(pred, labels)
                loss.backward()
                optimizer.step()

        pred_list = []
        feats_list = []

        for i, data in enumerate(self.dataset):
            imgs, labels = data 
            imgs, labels = imgs.cuda(), labels.cuda()

            optimizer.zero_grad()
            pred, feats = self.model(imgs)

            pred_list.append(pred)
            feats_list.append(feats)
            
            pred = pred.cuda()

            loss = crossEntropy(pred, labels) # +KLDiv(pred, labels)
            loss.backward()
            optimizer.step()

        print(pred_list[0:2])
        print(feats_list[0:2])

        self.model = self.model.to('cpu')

        self.model_dict = self.model.state_dict()
        torch.cuda.empty_cache()
        return TensorDataset(torch.Tensor(feats_list), torch.Tensor(pred_list))

    def get_data(self, key):
        return self.model_dict[key]


