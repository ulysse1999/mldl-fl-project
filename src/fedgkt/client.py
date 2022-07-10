from http import server
from torch.optim import SGD
from torch.nn import CrossEntropyLoss, KLDivLoss
from resnet8 import ResNet8
import torch
from torch.utils.data import Dataset, Subset, TensorDataset
import numpy as np
from customdataset import CustomDataset


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

    def __init__(self, index, normalization, local_dataset, batch_size=32, epochs=1):
        self.index = index
        self.model = ResNet8(normalization)
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

        kld_flag = 0

        self.model.cuda()
        self.model.train()

        pred_list = []
        feats_list = []
        labels_list = []

        if server_logit is not None:
            server_logit = server_logit[self.index].softmax(dim=1)
            dataset = CustomDataset(self.dataset.images, server_logit, self.dataset.targets) 
            kld_flag = 1
        else:
            dataset = self.dataset
        
        for epoch in range(self.epochs):
            for i, data in enumerate(dataset):
                if kld_flag == 0:
                    imgs, labels = data
                    imgs, labels = imgs.cuda(), labels.cuda()
                else:                    
                    imgs, s_logit, labels = data 
                    imgs, s_logit, labels = imgs.cuda(), s_logit.cuda(), labels.cuda()

                optimizer.zero_grad()
                pred, feats = self.model(imgs)
                
                if epoch == self.epochs-1:
                    pred_list.extend(pred)
                    feats_list.extend(feats)
                    labels_list.extend(labels)
                pred = pred.cuda()

                if kld_flag == 0:
                    loss = crossEntropy(pred,labels)
                else:
                    normalized_pred = pred.softmax(dim=1)
                    s_logit = s_logit.softmax(dim=1)

                    loss = crossEntropy(pred, labels) + KLDiv(normalized_pred.log(), s_logit)

                loss.backward()
                optimizer.step()


        self.model = self.model.to('cpu')

        self.model_dict = self.model.state_dict()
        torch.cuda.empty_cache()

        learnings = CustomDataset(feats_list, pred_list, labels_list)

        return learnings

    def get_data(self, key):
        return self.model_dict[key]


