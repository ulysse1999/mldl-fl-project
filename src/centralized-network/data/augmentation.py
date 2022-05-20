from torchvision.transforms import Compose, ToTensor, Normalize, RandomHorizontalFlip, RandomCrop


# CIFAR 10 dataset mean and sd for each channels :
#     mean: 0.49139968, 0.48215827 ,0.44653124
#     std: 0.24703233 0.24348505 0.26158768



def get_transform(p_flip=0.5):
    transform = Compose(
        [   
            RandomCrop(size=32, padding=4, padding_mode='reflect'),
            RandomHorizontalFlip(p_flip),
            ToTensor(),
            Normalize(mean=[0.491, 0.482, 0.446],
                                     std=[0.247, 0.243, 0.261])
        ]
    )

    return transform

