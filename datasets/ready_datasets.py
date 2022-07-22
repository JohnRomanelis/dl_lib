from .modelnet import ModelNet40Sampled, ModelNet40SampledCustom
from dl_lib.transforms import RandomPointDropout, RandomRotate, RandomShuffle, AnisotropicScale, ToTensor, UnitSphereNormalization
from dl_lib.transforms import TwoCropsTransform
from torch.utils.data import DataLoader


from .modelnet40c import ModelNet40C


def get_ModelNet40(path, name="original", batch_size=32, drop_last=False):
    # Args:
    #       - path: the path to modelnet40 data
    #       - name: the version of ModelNet to load
    #               Options:
    #                - "original" : the version used in dgcnn, PCT, pointMLP etc
    #                - "rotated"  : same augmentation as original but with rotation across all axises 
    #
    # Returns:  the train and validation dataloader

    assert name in ["original", "rotated", "normalized", "two_crops_original", "two_crops_rotated"]

    if name == "original":
        train_dataset = ModelNet40Sampled(path, num_points=1024, partition='train')
        valid_dataset = ModelNet40Sampled(path, num_points=1024, partition='test')

    if name == "normalized":
        train_transforms = [RandomPointDropout(),
                            RandomShuffle(),
                            UnitSphereNormalization(), 
                            #AnisotropicScale(), 
                            ToTensor()]

        valid_transforms = [UnitSphereNormalization(), 
                            ToTensor()]
        
        train_dataset = ModelNet40SampledCustom(path, num_points=1024, partition='train', transforms=train_transforms)
        valid_dataset = ModelNet40SampledCustom(path, num_points=1024, partition='test' , transforms=valid_transforms)

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

    elif name == "two_crops_original":

        train_transforms = TwoCropsTransform([
                                RandomPointDropout(), 
                                RandomShuffle(), 
                                AnisotropicScale(), 
                                ToTensor()])

        train_dataset = ModelNet40SampledCustom(path, num_points=1024, partition='train', transforms=train_transforms)
        valid_dataset = ModelNet40Sampled(path, num_points=1024, partition='test')

    elif name == "two_crops_rotated":

        train_transforms = TwoCropsTransform([
                                RandomPointDropout(), 
                                RandomShuffle(), 
                                AnisotropicScale(), 
                                ToTensor(),
                                RandomRotate(180, 0),
                                RandomRotate(180, 1),
                                RandomRotate(180, 2)])    

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


def get_ModelNet40C(path, corruption="all", severity="all", batch_size=32):
    """
    Loads ModelNet40-C dataset
     - You can either select a specific corruption type and severity 
       or ask for multiple versions.
    """ 

    # CONFIGURE CORRUPTION TYPE
    corruptions = {
        "Density" : ["occlusion", "lidar", "density", "density_inc", "cutout"],
        "Noise" : ["uniform", "gaussian", "impulse", "background", "upsampling"],
        "Transformation" : ["rotation", "shear", "distortion", "distortion_rbf", "distortion_rbf_inv"]
    }

    # select corruption by corruption category
    if corruption in ["Density", "Noise", "Transformation"]:
        corruption = corruptions[corruption]

    else: 
        # there is no need to separate corruptions by category
        corruptions = [*corruptions['Density'], *corruptions['Noise'], *corruptions['Transformation']]

        if corruption == "all":
            corruption = corruptions
        
        elif isinstance(corruption, (list, tuple)):
            for c in corruption:
                assert c in corruptions, f"'{c}' is not a valid corruption option"

        elif isinstance(corruption, str):
            assert corruption in corruptions, f"'{corruption}' is not a valid corruption option"
            corruption = [corruption]
        else:
            raise ValueError
    
    # CONFIGURE SEVERITY LEVER
    severity_levels = [1, 2, 3, 4, 5]

    if severity == "all":
        severity = severity_levels
    elif isinstance(severity, (list, tuple)):
        for s in severity:
            assert s in severity_levels, f"'{s}' is not a valid severity option"
    elif isinstance(severity, int):
        assert severity in severity_levels, f"'{severity}' is not a valid severity option"
        severity = [severity]
    else: 
        raise ValueError
    
    dataloaders = []

    for c in corruption:
        for s in severity:
            dataloaders.append(
                DataLoader(
                    ModelNet40C(path, corruption=c, severity=s),
                    batch_size=batch_size, shuffle=False, num_workers=8, drop_last=False
                )
            )
    
    if len(dataloaders) == 1:
        return dataloaders[0]
    
    return dataloaders
    
