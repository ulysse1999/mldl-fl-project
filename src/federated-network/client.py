from resnet50 import ResNet
import torch, torch.nn
import torchvision
from torch.optim import SGD
from torch.nn import CrossEntropyLoss


class Client:

    def __init__(self, normalization, local_dataset, batch_size=32 ,epochs=1):
        self.normalization = normalization
        self.dataset = torch.utils.data.DataLoader(
            local_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
            )
        self.epochs=epochs


    def train(self):

        optimizer = SGD(self.model.parameters(), lr=1)

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

        self.model_dict = self.model.state_dict()


    def get_data(self, key):
        return self.model_dict[key].data.clone()

    def set_model(self, model_dict):
        self.model = ResNet(self.normalization)
        self.model.load_state_dict(model_dict)

    