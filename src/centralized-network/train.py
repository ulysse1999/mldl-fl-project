from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam
from resnet50 import ResNet
from earlystopping import EarlyStopping
import ray
import numpy as np
from data.provider import get_testing_data

# for HP tuning : https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html#the-train-function 

def train(config, , trainloader, valloader, checkpoint_dir = "", n_epochs=10):
    """
    training of the global, centralized model

    validation accuracy at each epoch for raytune early stopping
    
    """

    model = ResNet()
    model.cuda()

    criterion = CrossEntropyLoss()
    criterion.cuda()


    optimizer = SGD(model.parameters(), lr=config["lr"], weight_decay = config["weightdecay"], momentum=0.9) if config["optimizer"]=="SGD" \
                else Adam(model.parameters, lr=config["lr"], weight_decay = config["weightdecay"])

    for i_epoch in range(n_epochs):
        
        for i, data in enumerate(trainloader):
            imgs, labels = data
            imgs, labels = imgs.cuda(), labels.cuda()

            optimizer.zero_grad()
            pred = model(imgs)
            pred = pred.cuda()
            
            loss = criterion(pred, labels)
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


def test_accuracy(model, device='cpu'):

    testset = get_testing_data()

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    
    return correct / total





