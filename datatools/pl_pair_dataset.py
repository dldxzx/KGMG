import os
import pickle
import lmdb
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm
import sys
sys.path.append('/home/user/fpk/KGMG')
from utils.data import PDBProtein, parse_sdf_file
from .pl_data import ProteinLigandData, torchify_dict


# class PocketLigandPairDataset(Dataset):

#     def __init__(self, raw_path, transform=None, version='final'):
#         super().__init__()
#         self.raw_path = raw_path.rstrip('/')
#         self.index_path = os.path.join(self.raw_path, 'index.pkl')
#         # self.processed_path = os.path.join(os.path.dirname(self.raw_path),
#         #                                    os.path.basename(self.raw_path) + f'_processed_{version}.lmdb')
#         # self.processed_path = os.path.join(os.path.dirname(self.raw_path), 'crossdocked_pocket10/' + os.path.basename(
#         #         self.raw_path) + '_processed.lmdb')
#         self.processed_path = '/home/user/fpk/SurfDM111/data/crossdocked_pocket10/crossdocked_pocket10_processed.lmdb'
#         self.transform = transform
#         self.db = None

#         self.keys = None

#         if not os.path.exists(self.processed_path):
#             print(f'{self.processed_path} does not exist, begin processing data')
#             self._process()

#     def _connect_db(self):
#         """
#             Establish read-only database connection
#         """
#         assert self.db is None, 'A connection has already been opened.'
#         self.db = lmdb.open(
#             self.processed_path,
#             map_size=10*(1024*1024*1024),   # 10GB
#             create=False,
#             subdir=False,
#             readonly=True,
#             lock=False,
#             readahead=False,
#             meminit=False,
#         )
#         with self.db.begin() as txn:
#             self.keys = list(txn.cursor().iternext(values=False))

#     def _close_db(self):
#         self.db.close()
#         self.db = None
#         self.keys = None
        
#     def _process(self):
#         db = lmdb.open(
#             self.processed_path,
#             map_size=10*(1024*1024*1024),   # 10GB
#             create=True,
#             subdir=False,
#             readonly=False,  # Writable
#         )
#         with open(self.index_path, 'rb') as f:
#             index = pickle.load(f)
#         # index = torch.load(self.index_path)['train']

#         num_skipped = 0
#         with db.begin(write=True, buffers=True) as txn:
#             for i, (pocket_fn, ligand_fn, *_) in enumerate(tqdm(index)):
#                 # print(os.path.join(self.raw_path, ligand_fn))
#                 # exit()
#                 if pocket_fn is None: continue
                
#                 try:
#                     # data_prefix = '/data/work/jiaqi/binding_affinity'
#                     data_prefix = self.raw_path
#                     # pocket_dict = PDBProtein(os.path.join(data_prefix, pocket_fn)).to_dict_atom()
#                     pocket_dict = PDBProtein(os.path.join(data_prefix, pocket_fn)).to_dict_atom()
#                     ligand_dict = parse_sdf_file(os.path.join(data_prefix, ligand_fn))
#                     data = ProteinLigandData.from_protein_ligand_dicts(
#                         protein_dict=torchify_dict(pocket_dict),
#                         ligand_dict=torchify_dict(ligand_dict),
#                     )
#                     data.protein_filename = pocket_fn
#                     data.ligand_filename = ligand_fn
#                     data = data.to_dict()  # avoid torch_geometric version issue
#                     txn.put(
#                         key=str(i).encode(),
#                         value=pickle.dumps(data)
#                     )
#                 except:
#                     num_skipped += 1
#                     print('Skipping (%d) %s' % (num_skipped, ligand_fn, ))
#                     continue
#         db.close()
    
#     def __len__(self):
#         if self.db is None:
#             self._connect_db()
#         return len(self.keys)

#     def __getitem__(self, idx):
#         data = self.get_ori_data(idx)
#         if self.transform is not None:
#             data = self.transform(data)
#         return data

#     def get_ori_data(self, idx):
#         if self.db is None:
#             self._connect_db()
#         key = self.keys[idx]
#         data = pickle.loads(self.db.begin().get(key))
#         data = ProteinLigandData(**data)
#         data.id = idx
#         assert data.protein_pos.size(0) > 0
#         return data


class PocketLigandPairDataset(Dataset):

    def __init__(self, raw_path, dataset='crossdock', transform=None):
        super().__init__()
        self.raw_path = raw_path.rstrip('/')
        if dataset == 'pdbind':
            self.file_path = './data/pdbind/'
            self.file_path = self.raw_path
            self.index_path = os.path.join(self.file_path, 'index.pkl')
            self.processed_path = os.path.join(self.file_path, 'pdbind_processed.lmdb')
            # print(self.processed_path)
            self.name2id_path = os.path.join(self.file_path, 'pdbind_name2id.pt')
        else:
            self.file_path = self.raw_path
            self.index_path = os.path.join(self.raw_path, 'index.pkl')  # crossdock 'crossdock_cutoff/'+
            # self.processed_path = os.path.join(os.path.dirname(self.raw_path), 'crossdocked_pocket10/' + os.path.basename(
            #     self.raw_path) + '_processed.lmdb')
            self.processed_path = '/home/user/fpk/SurfDM111/data/crossdocked_pocket10/crossdocked_pocket10_processed.lmdb'
            # self.name2id_path = os.path.join(os.path.dirname(self.raw_path),
            #                                  'crossdocked_pocket10/' + os.path.basename(self.raw_path) + '_name2id.pt')
            self.name2id_path = '/home/user/fpk/SurfDM111/data/crossdocked_pocket10/crossdocked_pocket10_name2id.pt'
        # self.name2id_path = os.path.join(os.path.dirname(self.raw_path), os.path.basename(self.raw_path) + '_name2id.pt')
        self.transform = transform
        self.db = None

        self.keys = None

        # print(self.processed_path)

        # print(os.path.exists(self.processed_path))
        if not os.path.exists(self.processed_path):
            self._process()
            self._precompute_name2id()
        if not os.path.exists(self.name2id_path):
            self._precompute_name2id()
        self.name2id = torch.load(self.name2id_path)
    def _connect_db(self):
        """
            Establish read-only database connection
        """
        assert self.db is None, 'A connection has already been opened.'
        self.db = lmdb.open(
            self.processed_path,
            map_size=10 * (1024 * 1024 * 1024),  # 10GB
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.db.begin() as txn:
            self.keys = list(txn.cursor().iternext(values=False))

    def _close_db(self):
        self.db.close()
        self.db = None
        self.keys = None

    def _process(self):
        db = lmdb.open(
            self.processed_path,
            map_size=10 * (1024 * 1024 * 1024),  # 10GB
            create=True,
            subdir=False,
            readonly=False,  # Writable
        )
        with open(self.index_path, 'rb') as f:
            index = pickle.load(f)

        num_skipped = 0
        with db.begin(write=True, buffers=True) as txn:
            for i, (pocket_fn, ligand_fn, _, rmsd_str) in enumerate(tqdm(index)):
                if pocket_fn is None: continue
                try:
                    ligand_dict = parse_sdf_file(os.path.join(self.raw_path, ligand_fn))
                    ligand_pos = ligand_dict['pos']
                    pocket_dict = PDBProtein(os.path.join(self.raw_path, pocket_fn)).to_dict_atom()
                    # pocket_dict = PDBProtein(os.path.join(self.raw_path, pocket_fn)).to_dict_atom_cutoff(ligand_pos,
                                                                                                        #  8.0)

                    data = ProteinLigandData.from_protein_ligand_dicts(
                        protein_dict=torchify_dict(pocket_dict),
                        ligand_dict=torchify_dict(ligand_dict),
                    )
                    data.protein_filename = pocket_fn
                    data.ligand_filename = ligand_fn
                    txn.put(
                        key=str(i).encode(),
                        value=pickle.dumps(data)
                    )
                except:
                    num_skipped += 1
                    print('Skipping (%d) %s' % (num_skipped, ligand_fn,))
                    continue
        db.close()

    def _precompute_name2id(self):
        name2id = {}
        for i in tqdm(range(self.__len__()), 'Indexing'):
            # if i<63340:
            #     continue
            try:
                data = self.__getitem__(i)
            except AssertionError as e:
                print(i, e)
                continue
            name = (data.protein_filename, data.ligand_filename)
            name2id[name] = i
        torch.save(name2id, self.name2id_path)

    def __len__(self):
        if self.db is None:
            self._connect_db()
        return len(self.keys)

    def __getitem__(self, idx):
        if self.db is None:
            self._connect_db()
        key = self.keys[idx]
        data = pickle.loads(self.db.begin().get(key))
        data.id = idx
        # if not data.protein_pos.size(0)>0:
        #     print(idx)
        #     print(key)
        assert data.protein_pos.size(0) > 0
        if self.transform is not None:
            data = self.transform(data)
        return data

        
if __name__ == '__main__':
    path = '/home/user/fpk/SurfDM111/data/crossdocked_pocket10'
    dataset = PocketLigandPairDataset(path)[89225]
    print(dataset)

