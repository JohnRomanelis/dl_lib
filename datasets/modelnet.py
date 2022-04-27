# Disclaimer: ModelNet40Sampled dataset is the version of the dataset used in dgcnn. 
#             The code is copied from the original dgcnn github repo. 
#             There are some minor changes to the code to receive a path for the dataset
# NOTES about the dataset and the augmentation procedure:
#           - Using a pre-sampled version of ModelNet40
#           - Does not apply any rotation to the input models that are pre-aligned 
#
# There is also a second version of the dataset, "ModelNet40SampledCustom". 
# This version receives custom transforms to apply to the input points.
# Note: the transforms should receive only the input pointcloud and not the labels.
# A series of such transforms as well as a custom collate function are also provided
# TODO: Move transforms and collate function to different folder.
#       They still remain in this file as they are dataset specific transforms  

import os
import glob
import h5py 
import numpy as np
from torch.utils.data import Dataset

def download(path=None):
    # adding the ability to use custom path
    if path is None:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        DATA_DIR = os.path.join(BASE_DIR, 'data')
    else:
        DATA_DIR = path

    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))

def load_data(partition, path=None):
    download(path)
    # adding the ability to use custom path
    if path is None:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        DATA_DIR = os.path.join(BASE_DIR, 'data')
    else: 
        DATA_DIR = path

    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5'%partition)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label

def random_point_dropout(pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''
    # for b in range(batch_pc.shape[0]):
    dropout_ratio = np.random.random()*max_dropout_ratio # 0~0.875    
    drop_idx = np.where(np.random.random((pc.shape[0]))<=dropout_ratio)[0]
    # print ('use random drop', len(drop_idx))

    if len(drop_idx)>0:
        pc[drop_idx,:] = pc[0,:] # set to the first point
    return pc

def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud

def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud


class ModelNet40Sampled(Dataset):
    def __init__(self, path, num_points, partition='train'):
        assert partition in ['train', 'test'], "Partition should be either 'train' or 'test'"
        self.data, self.label = load_data(partition, path)
        self.num_points = num_points
        self.partition = partition        

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points] 
        label = self.label[item]
        if self.partition == 'train':
            pointcloud = random_point_dropout(pointcloud) # open for dgcnn not for our idea for all
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]


class ModelNet40SampledCustom(ModelNet40Sampled):
    # A version of ModelNet40 sampled with custom transforms
    def __init__(self, path, num_points, partition='train', transforms=[]):
        super().__init__(path, num_points, partition)

        # making transforms a list to handle multiple transforms
        self.transforms = transforms if isinstance(transforms, (tuple, list)) else [transforms]
        # NOTE: transforms should operate only on the data, not the label

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        for t in self.transforms:
            pointcloud = t(pointcloud)
        return pointcloud, label       


#######################################
###                                 ###
###           Transforms            ###
###                                 ###
#######################################
# A series of transforms specifically desinged for the ModelNet40 dataset
import torch
import random # for RandomRotate transform
import math # for RandomRotate transform
import spconv.pytorch.utils as sputils


class VoxelizePointCloud:

    def __init__(self, voxel_size,
                       pc_coord_range,
                       num_point_features,
                       max_num_voxels,
                       max_num_points_per_voxel, 
                       keep_pos = False):


        self.voxel_gen = sputils.PointToVoxel(
            vsize_xyz=voxel_size, 
            coors_range_xyz = pc_coord_range,
            num_point_features = num_point_features,
            max_num_voxels = max_num_voxels,
            max_num_points_per_voxel = max_num_points_per_voxel
        )


        # there could also be an option to keep the original point coordinates
        # e.g. to create a point branch
        self.keep_pos = keep_pos


    def __call__(self, pc):
        # generating voxels (padding the empty points with the mean value of the voxel)
        voxels, coords, num_points_per_voxel = self.voxel_gen(pc, empty_mean=True)

        # changing the coordinates axises to be in the (x, y, z) order
        coords = coords[:, [2, 1, 0]]

        # creating a dictionary to return the voxelized data + metadata
        d = {
            "voxels": voxels, 
            "coords": coords,
            "num_points_per_voxel": num_points_per_voxel,
            "num_voxels": voxels.shape[0]
        }

        if self.keep_pos:
            d.update({
                "pos": pc
            })
        return d

class RandomShuffle:
    def __call__(self, pc):
        np.random.shuffle(pc)
        return pc

class AnisotropicScale:
    def __call__(self, pc):
        xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
        xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
        pointcloud = pc
        translated_pointcloud = pointcloud * xyz1 + xyz2
        return translated_pointcloud.astype(np.float32)

class RandomPointDropout:

    def __init__(self, max_dropout_ratio = 0.875):
        self.max_dropout_ratio = max_dropout_ratio
        
    def __call__(self, pc):  
        dropout_ratio = np.random.random() * self.max_dropout_ratio # 0 ~ 0.875
        drop_idx = np.where(np.random.random((pc.shape[0])) <= dropout_ratio)[0]

        if len(drop_idx) > 0:
            pc[drop_idx,:] = pc[0,:] # set to the first point

        return pc

class ToTensor:

    def __call__(self, pc):
        pc = torch.from_numpy(pc)
        return pc


class RandomRotate:

    def __init__(self, theta, axis, torch_or_np = "torch"):
        # Args: 
        #   - theta      : rotation angle (rotation will be from [-theta, theta])
        #   - axis       : the rotation axis (0,1,2 for x,y,z)
        #   - torch_or_np: whether the input pointcloud is represented by a torch.Tensor or np.Array

        self.theta = math.pi * theta / 180.0 # transforming angle to rads from degs
        self.axis = axis
        assert torch_or_np in ["torch", "np", "numpy"]
        self.use_torch = True if torch_or_np == "torch" else False

    def __call__(self, pc):
        # pc : a set of points with shape Nx3

        degree = random.uniform(-self.theta, self.theta)
        sin, cos = math.sin(degree), math.cos(degree)

        if self.axis == 0:
            matrix = [[1, 0, 0], [0, cos, sin], [0, -sin, cos]]
        elif self.axis == 1:
            matrix = [[cos, 0, -sin], [0, 1, 0], [sin, 0, cos]]
        else:
            matrix = [[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]]

        if self.use_torch:
            matrix = torch.tensor(matrix)
            pc = pc.unsqueeze(-1)
        else: 
            matrix = np.array(matrix)
            pc = pc[..., np.newaxis]

        return (matrix @ pc).squeeze(-1)


        

from collections import defaultdict
# custom collate function that can operate on voxels
def custom_collate_fn(batch_list):
    """
        This collate function is to be used for a classification task
        Each element of the batch_list is a dict or a tuple of dicts
        The following keys are to be concatenated
            - y : for labels
            - voxels 
            - coords
        The function should output a tuple (x_dict, y_dict)
    """
    
    # seperating the tuple in case of tuple input
    if isinstance(batch_list[0], tuple):
        # separating the input from the labels
        x = [batch_item[0] for batch_item in batch_list]
        y = [batch_item[1] for batch_item in batch_list]
    else:
        x = batch_list
        y = []
        
        
    x_dict_list = defaultdict(list)

    if isinstance(y[0], dict):
        y_dict_list = defaultdict(list)
    else:
        y_dict_list = []
    
    # creating a per key list instead of a list of dicts 
    # for x
    for i, b in enumerate(x):
        for k, v in b.items():
            x_dict_list[k].append(v)
        
        # adding a batch index
        batch_idx = i * torch.ones(b["voxels"].shape[0], 
                                    dtype=torch.int32)
        x_dict_list["batch_idx"].append(batch_idx)
        
        # adding a linear voxel index
        linear_idx = torch.arange(b["voxels"].shape[0], 
                                    dtype=torch.int32)
        x_dict_list["linear_idx"].append(linear_idx)
    

    # concatenating values
    # for x
    x_dict = {}

    for k, v in x_dict_list.items():
        if k in ["voxels", "coords", "batch_idx", 
                 "num_points_per_voxel", "linear_idx"]:
            x_dict[k] = torch.cat(v, dim=0)
        else: 
            x_dict[k] = v
        
    # for y
    if isinstance(y[0], dict):
        for i, b in enumerate(y):
            for k, v in b.items():
                y_dict_list[k].append(v)

        y_dict = {}            
        for k, v in y_dict_list.items():
            if k in ["y"]:
                y_dict[k] = torch.tensor(v)
            else:
                y_dict[k] = v

    else:
        # y is a list of numpy arrays 
        y = np.concatenate(y)
        y_dict = torch.from_numpy(y)
            

    x_dict["batch_size"] = len(batch_list)
            
    return (x_dict, y_dict)
