import spconv.pytorch.utils as sputils
import torch
import numpy as np

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

    if len(y) > 0:
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
    
    x_dict["batch_size"] = len(batch_list)

    # for y
    if len(y) > 0:
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
                        
        return (x_dict, y_dict)

    return x_dict