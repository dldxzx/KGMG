import torch
from torch.utils.data import Subset
from .pl_pair_dataset import PocketLigandPairDataset
from .pdbbind import PDBBindDataset


# def get_dataset(config, *args, **kwargs):
#     name = config.name
#     root = config.path
#     if name == 'pl':
#         dataset = PocketLigandPairDataset(root, *args, **kwargs)
#     elif name == 'pdbbind':
#         dataset = PDBBindDataset(root, *args, **kwargs)
#     else:
#         raise NotImplementedError('Unknown dataset: %s' % name)

#     if 'split' in config:
#         split = torch.load(config.split)
#         subsets = {k: Subset(dataset, indices=v) for k, v in split.items()}
#         return dataset, subsets
#     else:
#         return dataset

def get_dataset(config, *args, **kwargs):
    type = config.type
    root = config.path
    name = config.name
    if type == 'pl':
        dataset = PocketLigandPairDataset(root, name, *args, **kwargs)
    else:
        raise NotImplementedError('Unknown dataset: %s' % type)

    if 'split' in config:
        print(config.split)
        split_by_name = torch.load(config.split)
        
        split = {
            k: [dataset.name2id[n] for n in names if n in dataset.name2id]
            for k, names in split_by_name.items()
        }
        subsets = {k: Subset(dataset, indices=v) for k, v in split.items()}
        return dataset, subsets
    else:
        return dataset
