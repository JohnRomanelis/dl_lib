from .shapenet import ShapeNet
from .modelnet import ModelNet40Sampled, ModelNet40SampledCustom # to access transforms and collate use the modelnet.py file
from .modelnet40c import ModelNet40C
from .ready_datasets import get_ModelNet40, get_ModelNet40C