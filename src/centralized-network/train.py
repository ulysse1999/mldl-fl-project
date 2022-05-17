from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam
from resnet50 import ResNet
from earlystopping import EarlyStopping
import ray
import numpy as np

# for HP tuning : https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html#the-train-function 

def train(model, trainloader, valloader, n_epochs=10, batch_size=256):
    """
    training of the global, centralized model

    if the loss seems stagnant : early stopping
    validation accuracy at each epoch
    
    """

    model = ResNet()
    model.cuda()

    criterion = CrossEntropyLoss()

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

        
        
        val_loss = 0.
        val_steps = 0
        total = 0
        correct = 0

        for i,data in enumerate(valloader):
            with torch.no_grad():
                imgs, labels = data
                imgs, labels = imgs.cuda(), labels.cuda()

                out = model(img)
                _, predicted = torch.max(out.data, 1) # out : probabilities, we ignore the indexes

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=(val_loss / val_steps), accuracy=correct / total)

        
        
    
    return model


def test_accuracy():

    # TODO
    print("TODO")




