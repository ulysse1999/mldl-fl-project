from torchvision.transforms import Compose, ToTensor, Normalize, RandomHorizontalFlip, RandomCrop


def get_transform(p_flip=0.5):
    transform = Compose(
        [   
            RandomCrop(size=32),
            RandomHorizontalFlip(p_flip),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        ]
    )

    return transform

