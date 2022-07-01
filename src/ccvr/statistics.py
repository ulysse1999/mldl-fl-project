from torch_intermediate_layer_getter import IntermediateLayerGetter
import torch

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

    layers = {
        "conv5x.2.n3" : "feature_extraction"
    }

    features = dict()

    for index in client_subset:

        features[index] = {}

        getter = IntermediateLayerGetter(trained_models[index].model, layers, keep_output=False)

        for data in clients[index].dataset:
            imgs, labels = data

            for i in range(imgs):
                img, label = imgs[i], labels[i]
                res_layer, _ = getter(img)
                lay = res_layer["feature_extraction"]

                if label in features[index]:
                    features[index][label].append(lay)
                else:
                    features[index][label] = [lay]

    means, covs = dict(), dict()

    for index in client_subset:
        means[index] = dict()
        covs[index] = dict()
        for label in features[index]:
            means[index][label] = compute_mean(features[index][label])
            covs[index][label] = compute_cov(features[index][label], means[index][label])

    final_means, final_covs = {}, {}

    for label in range(0,10):
        for index in client_subset:
            

    



