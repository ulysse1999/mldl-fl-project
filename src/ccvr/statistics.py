from torchvision.models.feature_extraction import create_feature_extractor
import torch, torch.nn as nn

def statistics(clients, client_subset, trained_models):
    """
    clients : Client instances, useful to access data
    client_subset : indexes of the selected clients for this round
    trained_models : (index -> ResNet model) 
    """
    features = dict()

    features = {i:[] for i in range(10)}

    for index in client_subset:

        

        model = trained_models[index].model

        # saving model last FC layer
        save_fc = model.fc

        model.avgpool = nn.Identity()
        model.fc = nn.Identity()

        for data in clients[index].dataset:
            imgs, labels = data

            feats = model(imgs)

            for i in range(len(imgs)):

                _img, label = imgs[i], labels[i]

                lab = label.item()
                
                features[lab].append(feats[i])


        
                

        # set the model correctly
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        model.fc = save_fc

    print("Features extracted")

    print("Means/covs computed")

    final_means, final_covs = {}, {}
    n_samples = {}

    for label in range(0,10):
        nc = len(features[label])
        
        n_samples[label] = nc

        final_means[label] = torch.mean(torch.stack(features[label]), 0)
        final_covs[label] = torch.cov(torch.stack(features[label]))

    print("Result computed")

    return final_means, final_covs, n_samples
    



