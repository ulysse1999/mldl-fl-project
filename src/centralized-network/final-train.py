# network training loop
from data.provider import get_training_data, get_testing_data
from data.augmentation import get_transform

def main(optimizer, lr, weightdecay, normalization, n_epochs=50):

    assert optimizer in {"SGD", "Adam"}, "optimizer must be in \{SGD, Adam\}"
    assert normalization in {"group", "batch"}, "normalization must be in \{"







if __name__=='__main__':
    main()

