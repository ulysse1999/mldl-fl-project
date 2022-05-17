from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam
from resnet50 import ResNet
from earlystopping import EarlyStopping
import ray
import numpy as np


# tune : learning rate, weight decay, momentum optimizer

# for HP tuning : https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html#the-train-function 

def train(model, dataloader, n_epochs=10, batch_size=256):
    """
    training of the global, centralized model
    """

    model = ResNet()
    model.cuda()

    criterion = CrossEntropyLoss()

    early_stopping = EarlyStopping()
    optimizer = optimizer(model.params())
    # criterrion
    # optimizer
    for i_epoch in range(n_epochs):
        losses=list()
        for i, data in enumerate(dataloader):
            imgs, labels = data
            imgs, labels = imgs.cuda(), labels.cuda()

            optimizer.zero_grad()
            pred = model(imgs)
            pred = pred.cuda()
            
            loss = criterion(pred, labels)

            losses.append(loss.item())

            loss.backward()
            optimizer.step()

        epoch_loss = np.mean(np.array(losses))
        early_stopping(epoch_loss, model, i_epoch)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        
    #model.load_state_dict(torch.load(early_stopping.path))
    return model




