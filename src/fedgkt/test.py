import torch
import numpy as np
from data.provider import get_testing_data

# for HP tuning : https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html#the-train-function 


def test_accuracy(cl_model, s_model, transform=None, device='cpu'):

    testloader = get_testing_data(transform, BATCH_SIZE=10000)

    print("Testing model")

    cl_model.eval() # eval mode
    s_model.eval() # eval mode

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            cl_model.to(device)
            pred, feats = cl_model(images)
            feats = feats.to(device)
            s_model.to(device)
            output = s_model(feats)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test set accuracy (global model) : {correct/total}")
    
    return 





