from resnet50 import ResNet
import torch, torch.nn
import torchvision
from torch.optim import SGD
from torch.nn import CrossEntropyLoss


class Client:

    def __init__(self, normalization, local_dataset, epochs=1):
        self.model = ResNet(normalization)
        self.optimizer = SGD(lr=1)
        self.dataset = torch.utils.data.DataLoader(
            local_dataset,
            BATCH_SIZE=32,
            shuffle=True,
            num_workers=2,
            pin_memory=True
            )
        self.epochs=epochs


    def train(self):

        criterion = CrossEntropyLoss()
        criterion.cuda()

        for epoch in self.epochs:
            # training loop
            for i, data in enumerate(self.dataset):
                imgs, labels = data
                imgs, labels = imgs.cuda(), labels.cuda()

                self.optimizer.zero_grad()
                pred = self.model(imgs)
                pred = pred.cuda()
                
                loss = criterion(pred, labels)
                loss.backward()
                self.optimizer.step()

    def send_data():

    