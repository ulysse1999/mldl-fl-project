# network training loop
from data.provider import get_training_data, get_testing_data
from data.augmentation import get_transform
from resnet50 import ResNet
import torch, torch.nn as nn
import numpy as np
from torch.optim import SGD, Adam
from torch.nn import CrossEntropyLoss

# for now we choose not to save the model


def main(optimizer, lr, weightdecay, normalization, n_epochs=50):

    assert optimizer in {"SGD", "Adam"}, "optimizer must be in \{SGD, Adam\}"
    assert normalization in {"group", "batch"}, "normalization must be in \{batch, group\}"

    print(f"PARAMS :\n optimizer :{optimizer}\n lr :{lr}\n weight decay : {weightdecay}\n normalization :{normalization}")

    transform = get_transform()
    trainloader = get_training_data(transform=transform)

    model = ResNet(normalization)
    criterion = CrossEntropyLoss()
    model.cuda()
    criterion.cuda()
    

    optimizer = SGD(model.parameters(), lr=lr, weight_decay = weightdecay, momentum=0.9) if optimizer=="SGD" \
                else Adam(model.parameters(), lr=lr, weight_decay = weightdecay)

    model.train()

    # model training
    for epoch in range(n_epochs):
        print(f"epoch : {epoch}")
        losses = list()
        for i, data in enumerate(trainloader):
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
        print(f"Loss : {epoch_loss}")

    model.eval()
    
    testloader = get_testing_data()

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    
    print(f"Test set accuracy : {correct/total}")



if __name__=='__main__':
    # BN : Adam, lr=0.00035, weightdecay = 0.0225
    # GN : SGD, lr=0.00095, weightdecay = 0.0762
    main(optimizer='Adam', lr=0.00035, weightdecay=0.0225, normalization="batch")

