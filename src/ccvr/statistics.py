from torchvision.models.feature_extraction import create_feature_extractor
import torch, torch.nn as nn

def compute_mean(vectors):
    return sum(vectors) / len(vectors)

def compute_cov(vectors, mean):
    if len(vectors) == 1:
        return torch.zeros(vectors[0].shape)

    return sum([ torch.matmul(vec-mean, (vec-mean).t()) for vec in vectors]) / (len(vectors)-1) 



def statistics(clients, client_subset, trained_models):
    """
    clients : Client instances, useful to access data
    client_subset : indexes of the selected clients for this round
    trained_models : (index -> ResNet model) 
    """
    features = dict()

    for index in client_subset:

        features[index] = {}

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
                
                if label in features[index]:
                    features[index][label].append(feats[i])
                else:
                    features[index][label] = [feats[i]]

        # set the model correctly
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        model.fc = save_fc

    print("Features extracted")

    means, covs = dict(), dict()

    for index in client_subset:
        means[index] = dict()
        covs[index] = dict()
        for label in features[index]:
            means[index][label] = compute_mean(features[index][label])
            covs[index][label] = compute_cov(features[index][label], means[index][label])

    print("Means/covs computed")

    final_means, final_covs = {}, {}
    n_samples = {}

    print(features)

    for label in range(0,10):
        nc = sum([len(vectors) for index in client_subset for vectors in features[index][label] ])
        n_samples[label] = nc
        final_means[label] = sum([mean * len(vectors) for index in client_subset for vectors, mean in zip(features[index][label] ,means[index][label]) ]) / nc
        final_covs[label] = ( sum([cov * (len(vectors)-1) for index in client_subset for vectors, cov in zip(features[index][label] ,covs[index][label])]) \
            + sum([torch.matmul(means, mean.t()) * len(vectors) for index in client_subset for vectors, mean in zip(features[index][label] ,means[index][label])]) \
            - nc* torch.matmul(final_means[label], final_means[label].t() )) / (nc-1)

    print("Result computed")

    return final_means, final_covs, n_samples
    



