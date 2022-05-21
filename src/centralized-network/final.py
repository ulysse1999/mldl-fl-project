# network training loop
from data.provider import get_training_data, get_testing_data, get_train_validation_data
from data.augmentation import get_transform
from resnet50 import ResNet
import torch, torch.nn as nn
import numpy as np
from torch.optim import SGD, Adam
from torch.nn import CrossEntropyLoss
import torch
import torch.optim as optim
import argparse

# for now we choose not to save the model

CHECKPOINT_PATH = "checkpoint.pt"
path = CHECKPOINT_PATH


def test(normalization):

    best_trained_model = ResNet(normalization)

    model_state, _optimizer_state = torch.load(path)
    best_trained_model.load_state_dict(model_state)
    print("Model loaded")

    best_trained_model.eval()
    
    testloader = get_testing_data()

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to('cpu'), labels.to('cpu')
            outputs = best_trained_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    
    print(f"Test set accuracy : {correct/total}")


def train(optimizer, lr, weightdecay, normalization, n_epochs=50, model_path=None):

    assert optimizer in {"SGD", "Adam"}, "optimizer must be in \{SGD, Adam\}"
    assert normalization in {"group", "batch"}, "normalization must be in \{batch, group\}"

    print(f"PARAMS :\n optimizer :{optimizer}\n lr :{lr}\n weight decay : {weightdecay}\n normalization :{normalization}")

    transform = get_transform()
    trainloader, valloader = get_train_validation_data(transform=transform, train_proportion=0.9)

    model = ResNet(normalization)
    if model_path is not None:
        model_state, _ = torch.load(model_path)
        model.load_state_dict(model_state)
        
        
    model.cuda()

    criterion = CrossEntropyLoss()
    criterion.cuda()
    

    optimizer = SGD(model.parameters(), lr=lr, weight_decay = weightdecay, momentum=0.9) if optimizer=="SGD" \
                else Adam(model.parameters(), lr=lr, weight_decay = weightdecay)

    MAX_ACC = float("-inf")

    # model training
    for epoch in range(n_epochs):

        model.train()
        
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
        print(f"Training loss : {epoch_loss}")

        val_loss = 0.
        val_steps = 0
        total = 0
        correct = 0

        model.eval() #eval mode

        for i,data in enumerate(valloader):
            with torch.no_grad():
                imgs, labels = data
                imgs, labels = imgs.cuda(), labels.cuda()

                out = model(imgs)
                _, predicted = torch.max(out.data, 1) # out : probabilities, we ignore the values

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(out, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1

        val_acc = correct / total 

        print(f"Validation loss : {val_loss/val_steps}")
        print(f"Validation accuracy : {val_acc}")

        if val_acc > MAX_ACC:
            MAX_ACC = val_acc
            torch.save((model.state_dict(), optimizer.state_dict()), path)
            print("Model saved")
            
    
    


if __name__=='__main__':
    # from fist hyper parameter tuning : (with batch_size 256)
    # BN : Adam, lr=0.00035, weightdecay = 0.0225
    # GN : SGD, lr=0.00095, weightdecay = 0.0762
    # probably quite bad
    normalization="batch"
    # SET THE PARAMETERS FOR THE EXPERIMENT

    parser = argparse.ArgumentParser()
    parser.add_argument("--normalization", type=str, required=True, choices=["batch", "group"])
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--wd", type=float, required=True)
    parser.add_argument("--optimizer", type=str, required=True, choices=["SGD", "Adam"])
    parser.add_argument("--path", required=False, type=str, default=None) # for passing model
    parser.add_argument("--epochs", required=False, type=int, default=50)
    args = parser.parse_args()



    train(optimizer=args.optimizer, lr=args.lr, weightdecay=args.wd, normalization=args.normalization, model_path=args.path, n_epochs=args.epochs)
    test(args.normalization)

