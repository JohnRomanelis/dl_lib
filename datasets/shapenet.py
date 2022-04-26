import os
from torch.utils.data import Dataset
from torch_geometric.io import read_obj

class ShapeNet(Dataset):

    def __init__(self, root, category="all", transforms=[]):
        # storing the dataset root path
        self.root = root
        # storing the transforms
        self.transforms = transforms if isinstance(transforms, list) else [transforms]

        # getting the available shapenet categories and mapping them to intigers
        self.categories = self.read_categories()
        self.create_label_map()

        # creating a list to store the items of the dataset
        self.items = []

        if category == "all": 
            for cat in self.categories:
                self.read_category(cat)
        # read a specific category from shapenet or a list of categories
        else:
            if not isinstance(category, (list, tuple)):
                category = [category]
            for cat in category:
                assert cat in self.categories, "Not a valid category"
                self.read_category(cat)


    def read_category(self, category):
        # get the path of the said category
        category_path = os.path.join(self.root, category)
        # get the subfolder names
        subfolders = os.listdir(category_path)
        subfolders = [os.path.join(category, subf) for subf in subfolders]
        self.items.extend(subfolders)
    
    def read_categories(self):
        # reading available categories from subdirectories
        # and storing the alphabetically
        categories = os.listdir(self.root)
        categories.sort()
        return categories

    def create_label_map(self):
        # mapping every category to an intiger 
        # also creating an inverse dir to map intigers to the original labels
        self.label_map = {}
        self.inv_label_map = {}
        for i, cat in enumerate(self.categories):
            self.label_map[cat] = i
            self.inv_label_map[str(i)] = cat

    # read the number of examples
    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        category = item.split("/")[0] 

        # reading the model
        model_path = os.path.join(self.root, item, "model.obj")
        model = read_obj(model_path)

        # applying transforms to the model(mesh/pointcloud)
        for t in self.transforms:
            model = t(model)
        
        # adding the label information
        model.label = self.label_map[category]

        return model