from torchvision.models.feature_extraction import create_feature_extractor
import torch, torch.nn as nn

def statistics(clients, client_subset, trained_models):
    """
    clients : Client instances, useful to access data
    client_subset : indexes of the selected clients for this round
    trained_models : (index -> ResNet model) 
    """
    features = dict()

    for index in client_subset:

        features = {i:[] for i in range(10)}

        model = trained_models[index].model

        # saving model last FC layer
        save_fc = model.fc

        model.avgpool = nn.Identity()
        model.fc = nn.Identity()

        c = 0

        for data in clients[index].dataset:
            imgs, labels = data

            feats = model(imgs)

            for i in range(len(imgs)):

                _img, label = imgs[i], labels[i]

                lab = label.item()
                
                features[lab].append(feats[i])
                c+=1

        
                

        # set the model correctly
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        model.fc = save_fc

    print("Features extracted")

    print("Means/covs computed")

    final_means, final_covs = {}, {}
    n_samples = {}

    for label in range(0,10):
        nc = sum([len(features[index][label]) for index in client_subset ])
        print(f"nc = {nc}")
        n_samples[label] = nc

        final_means[label] = torch.mean(torch.Tensor(features[label]))
        final_covs[label] = torch.cov(torch.Tensor(features[label]))

    print("Result computed")

    return final_means, final_covs, n_samples
    



