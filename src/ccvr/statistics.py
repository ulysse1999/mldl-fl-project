from torch_intermediate_layer_getter import IntermediateLayerGetter

def statistics(clients, client_subset, trained_models):
    """
    clients : Client instances, useful to access data
    client_subset : indexes of the selected clients for this round
    trained_models : (index -> ResNet model) 
    """

    layers = {
        "conv5x.2.n3" : "feature_extraction"
    }

    for index in client_subset:

        getter = IntermediateLayerGetter(trained_models[index].model, layers, keep_output=False)
        res_layer, _ = getter(clients[index].dataset)

        for data in clients[index].dataset:
            res_layer, _ = getter(data)
            lay = res_layer["feature_extraction"]


