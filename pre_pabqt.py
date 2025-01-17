import os
import subprocess
import argparse
import os
import shutil
import logging
from tqdm import tqdm 

import numpy as np
import torch
from  torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score
from torch.nn.utils import clip_grad_norm_
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from tqdm.auto import tqdm

import utils.misc as misc
import utils.train as utils_train
import utils.transforms as trans
from datatools import get_dataset
from datatools.pl_data import FOLLOW_BATCH
from utils.transforms import *
from KGMG.model.KGMG import SurfDM

device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_config = '/home/user/fpk/KGMG/configs/training.yml'
# Load configs
config = misc.load_config(train_config)
config_name = os.path.basename(train_config)[:os.path.basename(train_config).rfind('.')]
misc.seed_all(config.train.seed)

# Transforms
protein_featurizer = trans.FeaturizeProteinAtom()
ligand_featurizer = trans.FeaturizeLigandAtom(config.data.transform.ligand_atom_mode)

transform_list = [
    protein_featurizer,
    ligand_featurizer,
    FeaturizeLigandBond(),
]
if config.data.transform.random_rot:
    transform_list.append(RandomRotation())
transform = Compose(transform_list)

# Datasets and loaders
dataset, subsets = get_dataset(
    config=config.data,
    transform=transform
)

train_set, val_set = subsets['train'], subsets['test']
print(f'Training: {len(train_set)} Validation: {len(val_set)}')
    # follow_batch = ['protein_element', 'ligand_element']
collate_exclude_keys = ['ligand_nbh_list']

val_loader = DataLoader(val_set, config.train.batch_size, shuffle=False,
                        follow_batch=FOLLOW_BATCH, exclude_keys=collate_exclude_keys)
protein_root = '/home/user/fpk/KGMG/data/crossdocked_pocket10'
protein_file = []
for idx, data in enumerate(val_loader):
    # protein_file = os.path.join(protein_root, data.protein_filename)
    print(data.protein_filename)
    protein_file.append(data.protein_filename)
protein_filename = []
for idx, pro_file in enumerate(protein_file):
    for file in pro_file:
        protein_filename.append(os.path.join(protein_root,file))

print(protein_filename)
pdb_dir = '/home/user/fpk/KGMG/examples/'
pdb_dir = protein_filename
output_dir = '/home/user/fpk/KGMG/data/test_pdbqt'

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)
# 遍历 PDB 文件
for pdb_file in pdb_dir:
    if pdb_file.endswith('.pdb'):
        # input_path = os.path.join(pdb_dir, pdb_file)
        # output_file = pdb_file.replace('.pdb', '.pdbqt')
        # output_path = os.path.join(output_dir, output_file)
        input_path = pdb_file  # 使用当前的 PDB 文件路径
        output_file = os.path.basename(pdb_file).replace('.pdb', '.pdbqt')  # 获取文件名并替换后缀
        output_path = os.path.join(output_dir, output_file)

        # 构造命令
        command = [
            '/path/bin/python', 
            '/path/lib/python3.8/site-packages/AutoDockTools/Utilities24/prepare_receptor4.py',
            '-r', input_path,
            '-o', output_path
        ]

        # 执行命令
        subprocess.run(command, check=True)
        print(f'Converted {pdb_file} to {output_file}')
       
# destination_folder = '/home/user/fpk/test_pdb'

# # 创建目标文件夹（如果不存在）
# os.makedirs(destination_folder, exist_ok=True)

# # 遍历 val_loader 以获取每个蛋白质文件的完整路径
# for idx, data in enumerate(val_loader):
#     for file in data.protein_filename:
#         full_path = os.path.join(protein_root, file)
#         protein_filename.append(full_path)
#         print(full_path)  # 打印完整路径

#         # 将文件复制到目标文件夹
#         shutil.copy(full_path, destination_folder)    


# print(len(protein_filename))