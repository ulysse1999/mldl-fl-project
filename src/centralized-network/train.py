

# tune : learning rate, weight decay, momentum optimizer

def train(model, dataloader, n_epochs):
    """
    training of the global, centralized model
    """

    print("IF CUDA IS NOT AVAILABLE, NOT EVEN WORTH TRYING TO RUN THE TRAINING")

    # criterrion
    # optimizer
    for i_epoch in range(n_epochs):
        for i, data in enumerate(dataloader):
            imgs, labels = data
            imgs, labels = imgs.cuda(), labels.cuda()

            optimizer.zero_grad()
            pred = model(imgs)
            pred = pred.cuda()

            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()





