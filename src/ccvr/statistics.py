from torchvision.models.feature_extraction import create_feature_extractor
import torch, torch.nn as nn

def compute_mean(vectors):
    return sum(vectors) / len(vectors)

def compute_cov(vectors, mean):
    if len(vectors) == 1:
        return torch.zeros(vectors[0].shape)

    return sum([ torch.matmul(vec-mean, (vec-mean).t()) for vec in vectors]) / (len(vectors)-1) 

def get_feature_shape(features):
    for k in features:
        for i in features[k]:
            if len(features[k][i]) >= 1:
                return features[k][i][0].shape




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

        c = 0

        for data in clients[index].dataset:
            imgs, labels = data

            feats = model(imgs)

            print(feats)

            features[index] = {i:[] for i in range(10)}
            
            for i in range(len(imgs)):
                _img, label = imgs[i], labels[i]

                lab = label.item()
                
                features[index][lab].append(feats[i])
                c+=1

        print(f"n data : {c}")
                

        # set the model correctly
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        model.fc = save_fc

    print("Features extracted")

    print("features nb :", {k:{i : len(features[k][i]) for i in features[k]} for k in features})

    shape = get_feature_shape(features)

    means, covs = dict(), dict()

    for index in client_subset:
        means[index] = dict()
        covs[index] = dict()
        for label in features[index]:
            means[index][label] = compute_mean(features[index][label]) if len(features[index][label])>=1 else torch.zeros(shape)
            covs[index][label] = compute_cov(features[index][label], means[index][label]) if len(features[index][label])>=1 else torch.matmul(torch.zeros(shape), torch.zeros(shape).t())

    print("Means/covs computed")

    final_means, final_covs = {}, {}
    n_samples = {}

    for label in range(0,10):
        nc = sum([len(features[index][label]) for index in client_subset ])
        print(f"nc = {nc}")
        n_samples[label] = nc
        final_means[label] = sum([means[index][label] * len(features[index][label]) for index in client_subset  ]) / nc
        final_covs[label] = ( sum([covs[index][label] * (len(features[index][label])-1) for index in client_subset ]) \
            + sum([torch.matmul(means[index][label], means[index][label].t()) * len(features[index][label]) for index in client_subset  ]) \
            - nc* torch.matmul(final_means[label], final_means[label].t() )) / (nc-1)

    print("Result computed")

    return final_means, final_covs, n_samples
    



