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
from model.KGMG import SurfDM

log_dir = '/home/user/fpk/KGMG/log'
writer = SummaryWriter(log_dir)
def log():

    # creat logger 
    logger = logging.getLogger(__name__)
    # log level as INFO
    logger.setLevel(logging.INFO)
    # 创建一个控制台处理器并设置格式
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    # 将处理器添加到 logger 中
    logger.addHandler(handler)
    return logger
def get_auroc(y_true, y_pred, feat_mode):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    avg_auroc = 0.
    possible_classes = set(y_true)
    for c in possible_classes:
        auroc = roc_auc_score(y_true == c, y_pred[:, c])
        avg_auroc += auroc * np.sum(y_true == c)
        mapping = {
            'basic': trans.MAP_INDEX_TO_ATOM_TYPE_ONLY,
            'add_aromatic': trans.MAP_INDEX_TO_ATOM_TYPE_AROMATIC,
            'full': trans.MAP_INDEX_TO_ATOM_TYPE_FULL
        }
        print(f'atom: {mapping[feat_mode][c]} \t auc roc: {auroc:.4f}')
    return avg_auroc / len(y_true)


if __name__ == '__main__':
    logger = log()
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

     # follow_batch = ['protein_element', 'ligand_element']
    collate_exclude_keys = ['ligand_nbh_list']
    train_iterator = utils_train.inf_iterator(DataLoader(
        train_set,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
        follow_batch=FOLLOW_BATCH,
        exclude_keys=collate_exclude_keys
    ))

    val_loader = DataLoader(val_set, config.train.batch_size, shuffle=False,
                            follow_batch=FOLLOW_BATCH, exclude_keys=collate_exclude_keys)
    
    pre_train = False
    if pre_train:
        pre_ckpt = '/home/user/fpk/KGMG/checkpoint/kgmg.pt'
        ckpt = torch.load(pre_ckpt, map_location=device)
        print(ckpt['config'].model)
        # Load model
        model = SurfDM(
            ckpt['config'].model,
            protein_atom_feature_dim=protein_featurizer.feature_dim,
            ligand_atom_feature_dim=ligand_featurizer.feature_dim
        ).to('cuda')
        model.load_state_dict(ckpt['model'], strict=False if 'train_config' in config.model else True)
        logger.info('Successfully load the model, iteration:{}'.format(ckpt['iteration']))
    else:
        model = SurfDM(
        config.model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim
    ).to(device)
    
    print(f'protein feature dim: {protein_featurizer.feature_dim} ligand feature dim: {ligand_featurizer.feature_dim}')
    logger.info(f'# trainable parameters: {misc.count_parameters(model) / 1e6:.4f} M')

    # Optimizer and scheduler
    optimizer = utils_train.get_optimizer(config.train.optimizer, model)
    scheduler = utils_train.get_scheduler(config.train.scheduler, optimizer)
    l1_lambda = 0.01
    def train(it):
        model.train()
        optimizer.zero_grad()
        for _ in range(config.train.n_acc_batch):
            batch = next(train_iterator).to(device)
            protein_noise = torch.randn_like(batch.protein_pos) * config.train.pos_noise_std
            gt_protein_pos = batch.protein_pos + protein_noise
            results = model(
                protein_pos=gt_protein_pos,
                protein_v=batch.protein_atom_feature.float(),
                batch_protein=batch.protein_element_batch,

                ligand_pos=batch.ligand_pos,
                ligand_v=batch.ligand_atom_feature_full,
                batch_ligand=batch.ligand_element_batch,
                
            )

            loss, loss_pos, loss_v = results['loss'], results['loss_pos'], results['loss_v']
            loss = loss / config.train.n_acc_batch
            # l1_norm = sum(p.abs().sum() for p in model.parameters())
            # total_loss = loss + l1_lambda * l1_norm
            loss.backward()
            
        orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
        optimizer.step()

        if it % 200 == 0:
            logger.info(
                '[Train] Iter %d | Loss %.6f (pos %.6f | v %.6f) | Lr: %.8f | Grad Norm: %.6f' % (
                    it, loss, loss_pos, loss_v, optimizer.param_groups[0]['lr'], orig_grad_norm
                )
            )
            for k, v in results.items():
                if torch.is_tensor(v) and v.squeeze().ndim == 0:
                    writer.add_scalar(f'train/{k}', v, it)
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
            writer.add_scalar('train/grad', orig_grad_norm, it)
            writer.flush()


    def validate(it):
        # fix time steps
        sum_loss, sum_loss_pos, sum_loss_v, sum_n = 0, 0, 0, 0
        sum_loss_bond, sum_loss_non_bond = 0, 0
        all_pred_v, all_true_v = [], []
        all_pred_bond_type, all_gt_bond_type = [], []
        with torch.no_grad():
            model.eval()
            for batch in tqdm(val_loader, desc='Validate'):
                batch = batch.to(device)
                batch_size = batch.num_graphs
                t_loss, t_loss_pos, t_loss_v = [], [], []
                for t in np.linspace(0, model.num_timesteps - 1, 10).astype(int):
                    time_step = torch.tensor([t] * batch_size).to(device)
                    results = model(
                        protein_pos=batch.protein_pos,
                        protein_v=batch.protein_atom_feature.float(),
                        batch_protein=batch.protein_element_batch,

                        ligand_pos=batch.ligand_pos,
                        ligand_v=batch.ligand_atom_feature_full,
                        batch_ligand=batch.ligand_element_batch,time_step=time_step,
                    )
                    loss, loss_pos, loss_v = results['loss'], results['loss_pos'], results['loss_v']

                    sum_loss += float(loss) * batch_size
                    sum_loss_pos += float(loss_pos) * batch_size
                    sum_loss_v += float(loss_v) * batch_size
                    sum_n += batch_size
                    all_pred_v.append(results['ligand_v_recon'].detach().cpu().numpy())
                    all_true_v.append(batch.ligand_atom_feature_full.detach().cpu().numpy())

        avg_loss = sum_loss / sum_n
        avg_loss_pos = sum_loss_pos / sum_n
        avg_loss_v = sum_loss_v / sum_n
        atom_auroc = get_auroc(np.concatenate(all_true_v), np.concatenate(all_pred_v, axis=0),
                               feat_mode=config.data.transform.ligand_atom_mode)

        if config.train.scheduler.type == 'plateau':
            scheduler.step(avg_loss)
        elif config.train.scheduler.type == 'warmup_plateau':
            scheduler.step_ReduceLROnPlateau(avg_loss)
        else:
            scheduler.step()

        logger.info(
            '[Validate] Iter %05d | Loss %.6f | Loss pos %.6f | Loss v %.6f e-3 | Avg atom auroc %.6f' % (
                it, avg_loss, avg_loss_pos, avg_loss_v * 1000, atom_auroc
            )
        )
        writer.add_scalar('val/loss', avg_loss, it)
        writer.add_scalar('val/loss_pos', avg_loss_pos, it)
        writer.add_scalar('val/loss_v', avg_loss_v, it)
        writer.flush()
        return avg_loss


    try:
        best_loss, best_iter = 100.0, None
        ckpt_dir = '/home/user/fpk/KGMG/checkpoint'
        for it in range(1, config.train.max_iters + 1):
            if pre_train:
                it+=338600
            # with torch.autograd.detect_anomaly():
            # if it > 0:
            #     config.train.optimizer.type = 'sgd'
            #     config.train.optimizer.lr = 1.e-3
            #     # Optimizer and scheduler
            #     optimizer = utils_train.get_optimizer(config.train.optimizer, model)
            #     scheduler = utils_train.get_scheduler(config.train.scheduler, optimizer)
            train(it)
            if it % config.train.val_freq == 0 or it == config.train.max_iters:
                val_loss = validate(it)
                if best_loss is None or val_loss < best_loss:
                    logger.info(f'[Validate] Best val loss achieved: {val_loss:.8f}')
                    best_loss, best_iter = val_loss, it
                    ckpt_path = os.path.join(ckpt_dir, '%d.pt' % it)
                    torch.save({
                        'config': config,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'iteration': it,
                    }, ckpt_path)
                else:
                    logger.info(f'[Validate] Val loss is not improved. '
                                f'Best val loss: {best_loss:.8f} at iter {best_iter}')
    except KeyboardInterrupt:
        logger.info('Terminating...')