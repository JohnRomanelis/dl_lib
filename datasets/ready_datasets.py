from .modelnet import ModelNet40Sampled, ModelNet40SampledCustom
from .modelnet import RandomPointDropout, RandomRotate, RandomShuffle, AnisotropicScale, ToTensor
from torch.utils.data import DataLoader


def get_ModelNet40(path, name="original", batch_size=32, drop_last=False):
    # Args:
    #       - path: the path to modelnet40 data
    #       - name: the version of ModelNet to load
    #               Options:
    #                - "original" : the version used in dgcnn, PCT, pointMLP etc
    #                - "rotated"  : same augmentation as original but with rotation across all axises 
    #
    # Returns:  the train and validation dataloader

    assert name in ["original"]

    if name == "original":
        train_dataset = ModelNet40Sampled(path, num_points=1024, partition='train')
        valid_dataset = ModelNet40Sampled(path, num_points=1024, partition='test')

    elif name == "rotated":
        train_transforms = [RandomPointDropout(), 
                            RandomShuffle(), 
                            AnisotropicScale(), 
                            ToTensor(),
                            RandomRotate(180, 0),
                            RandomRotate(180, 1),
                            RandomRotate(180, 2)]
        
        valid_transforms = [RandomShuffle(), 
                            ToTensor(),
                            RandomRotate(180, 0),
                            RandomRotate(180, 1),
                            RandomRotate(180, 2)]

        train_dataset = ModelNet40SampledCustom(path, num_points=1024, partition='train', transforms=train_transforms)
        valid_dataset = ModelNet40SampledCustom(path, num_points=1024, partition='test' , transforms=valid_transforms)


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=drop_last)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=False)

    return train_loader, valid_loader

