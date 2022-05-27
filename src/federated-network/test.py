import torch
import numpy as np
from data.provider import get_testing_data

# for HP tuning : https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html#the-train-function 


def test_accuracy(model, transform=None, device='cpu'):

    testloader = get_testing_data(transform)

    model.eval() # eval mode

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    
    return correct / total





