import argparse
import pickle
import os
from pathlib import Path
from rdkit.Chem import Draw
import numpy as np
from rdkit import Chem
from rdkit import RDLogger
import torch
from tqdm.auto import tqdm
from glob import glob
from collections import Counter
import sys
sys.path.append('/home/user/fpk/KGMG')
from utils.evaluation import eval_atom_type, scoring_func, analyze, eval_bond_length
from utils import misc, reconstruct, transforms
from utils.evaluation.docking_qvina import QVinaDockingTask
from utils.evaluation.docking_vina import VinaDockingTask
from rdkit.Chem.Descriptors import MolLogP, qed
from eval.sascorer import compute_sa_score, obey_lipinski
from eval.docking_2 import calculate_qvina2_score
from utils.reconstruct_mdm import make_mol_openbabel
from eval.diversity import calculate_diversity

def print_dict(d, logger):
    for k, v in d.items():
        if v is not None:
            logger.info(f'{k}:\t{v:.4f}')
        else:
            logger.info(f'{k}:\tNone')


def print_ring_ratio(all_ring_sizes, logger):
    for ring_size in range(3, 10):
        n_mol = 0
        for counter in all_ring_sizes:
            if ring_size in counter:
                n_mol += 1
        logger.info(f'ring size: {ring_size} ratio: {n_mol / len(all_ring_sizes):.3f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('sample_path', type=str)
    parser.add_argument('--verbose', type=eval, default=False)
    parser.add_argument('--eval_step', type=int, default=-1)
    parser.add_argument('--eval_num_examples', type=int, default=None)
    parser.add_argument('--save', type=eval, default=True)
    parser.add_argument('--protein_root', type=str, default='./data/crossdocked_pocket10')
    parser.add_argument('--atom_enc_mode', type=str, default='add_aromatic')
    parser.add_argument('--docking_mode', type=str, choices=['qvina', 'vina_score', 'vina_dock', 'none'])
    parser.add_argument('--exhaustiveness', type=int, default=20)
    args = parser.parse_args()

    # result_path = os.path.join(args.sample_path, 'eval_results')
    # os.makedirs(result_path, exist_ok=True)
    result_path = os.path.join(os.path.expanduser('~'), 'outputs')
    os.makedirs(result_path, exist_ok=True)
    logger = misc.get_logger('evaluate', log_dir=result_path)
    if not args.verbose:
        RDLogger.DisableLog('rdApp.*')

    # Load generated data
    results_fn_list = glob(os.path.join('/home/user/fpk/KGMG/outputs', '*result_*.pt'))
    results_fn_list = sorted(results_fn_list, key=lambda x: int(os.path.basename(x)[:-3].split('_')[-1]))
    if args.eval_num_examples is not None:
        results_fn_list = results_fn_list[:args.eval_num_examples]
    num_examples = len(results_fn_list)
    logger.info(f'Load generated data done! {num_examples} examples in total.')

    num_samples = 0
    all_mol_stable, all_atom_stable, all_n_atom = 0, 0, 0
    n_recon_success, n_eval_success, n_complete = 0, 0, 0
    results = []
    all_pair_dist, all_bond_dist = [], []
    all_atom_types = Counter()
    success_pair_dist, success_atom_types = [], Counter()
    pockt_mols = []
    times = []
    smiles_list = []
    high_affinity = 0

    with open('/home/user/fpk/KGMG/eval/test_vina_crossdock_dict.pkl', 'rb') as f:
        test_vina_score_list = pickle.load(f)
    test_vian_scores = list(test_vina_score_list.values())
    for example_idx, r_name in enumerate(tqdm(results_fn_list, desc='Eval')):
        r = torch.load(r_name)  # ['data', 'pred_ligand_pos', 'pred_ligand_v', 'pred_ligand_pos_traj', 'pred_ligand_v_traj']
        all_pred_ligand_pos = r['pred_ligand_pos_traj']  # [num_samples, num_steps, num_atoms, 3]
        all_pred_ligand_v = r['pred_ligand_v_traj']
        time = r['time']
        num_samples += len(all_pred_ligand_pos)
        ligand_name = r['data'].ligand_filename[:-4]

        pockt_mol = []
        for sample_idx, (pred_pos, pred_v) in enumerate(zip(all_pred_ligand_pos, all_pred_ligand_v)):
            pred_pos, pred_v = pred_pos[args.eval_step], pred_v[args.eval_step]
            # stability check
            pred_atom_type = transforms.get_atomic_number_from_index(pred_v, mode=args.atom_enc_mode)
            all_atom_types += Counter(pred_atom_type)
            r_stable = analyze.check_stability(pred_pos, pred_atom_type)
            all_mol_stable += r_stable[0]
            all_atom_stable += r_stable[1]
            all_n_atom += r_stable[2]

            pair_dist = eval_bond_length.pair_distance_from_pos_v(pred_pos, pred_atom_type)
            all_pair_dist += pair_dist
            protein_root = './data/test_pdbqt'
            sdf_dir = '/home/user/fpk/KGMG/sample_all'
            try:
                pred_aromatic = transforms.is_aromatic_from_index(pred_v, mode=args.atom_enc_mode)
                # mol = reconstruct.reconstruct_from_generated(pred_pos, pred_atom_type, pred_aromatic)
                mol = make_mol_openbabel(pred_pos, pred_atom_type)
                smiles = Chem.MolToSmiles(mol)
                smiles_list.append(smiles)
            # except reconstruct.MolReconsError:
            except (Chem.AtomValenceException, RuntimeError) as e:
                if args.verbose:
                    logger.warning('Reconstruct failed %s' % f'{example_idx}_{sample_idx}')
                continue
            n_recon_success += 1
            if '.' in smiles:
                continue
            n_complete += 1
            try:
                chem_results = scoring_func.get_chem(mol)
                receptor_file = os.path.basename(r['data'].protein_filename).replace('.pdb','')+'.pdbqt'
                receptor_file = Path(os.path.join(protein_root,receptor_file))
                g_vina_score = calculate_qvina2_score(receptor_file, mol, sdf_dir, return_rdmol=False,index=sample_idx )[0]
                print(f'example_idx:{example_idx} sample_idx:{sample_idx} smiles:{smiles} docking score: {g_vina_score}')
                if g_vina_score > -2:
                        continue
                vina_results = {
                    'score_only': g_vina_score
                }
                
                pockt_mol.append(mol)
                n_eval_success += 1
            except:
                if args.verbose:
                    logger.warning('Evaluation failed for %s' % f'{example_idx}_{sample_idx}')
                continue
            # if g_vina_score > -2:
            #         continue
            print("Generate vina score:", g_vina_score)
            rd_vina_score = test_vian_scores[example_idx]
            print('Reference vina score:', rd_vina_score)
            g_high_affinity = False
            if g_vina_score < rd_vina_score:
                high_affinity += 1.0
                g_high_affinity = True
            results.append({
                        'mol': mol,
                        'smiles': smiles,
                        'ligand_filename': r['data'].ligand_filename,
                        'pred_pos': pred_pos,
                        'pred_v': pred_v,
                        'chem_results': chem_results,
                        'vina': vina_results,
                        'high_affintiy': g_high_affinity
                    })
        times.append(time)
        pockt_mols.append(pockt_mol)
    print(n_recon_success, n_complete,n_eval_success)
    torch.save(smiles_list,'generate_all_smiles.pt')
    logger.info(f'Evaluate done! {num_samples} samples in total.')
    logger.info('Number of reconstructed mols: %d, complete mols: %d, evaluated mols: %d' % (
        n_recon_success, n_complete, len(results)))
    qed = [r['chem_results']['qed'] for r in results]
    sa = [r['chem_results']['sa'] for r in results]
    logp = [r['chem_results']['logp'] for r in results]
    lipinski = [r['chem_results']['lipinski'] for r in results]
    vina = [r['vina']['score_only'] for r in results]
    high_affinity_ratio = high_affinity / len(sa)
    logger.info('high_affinity_ratio: %.3f ' % (high_affinity_ratio))
    per_pocket_diversity = []
    for pkt_mol in pockt_mols:
        diversity = calculate_diversity(pkt_mol)
        per_pocket_diversity.append(diversity)
    per_pocket_diversity = [x for x in per_pocket_diversity if x != 0]
    logger.info('QED:   Mean: %.3f Median: %.3f std: %.2f' % (np.mean(qed), np.median(qed), np.std(qed)))
    logger.info('SA:    Mean: %.3f Median: %.3f std: %.2f' % (np.mean(sa), np.median(sa), np.std(sa)))
    logger.info('logp:    Mean: %.3f Median: %.3f std: %.2f' % (np.mean(logp), np.median(logp), np.std(logp)))
    logger.info('lipinski:    Mean: %.3f Median: %.3f std: %.2f' % (np.mean(lipinski), np.median(lipinski), np.std(lipinski)))
    logger.info('Vina Score:    Mean: %.3f Median: %.3f std: %.2f' % (np.mean(vina), np.median(vina), np.std(vina)))
    logger.info('per_pocket_diversity:    Mean: %.3f Median: %.2f std: %.2f' % (np.mean(per_pocket_diversity), np.median(per_pocket_diversity), np.std(per_pocket_diversity)))
    logger.info('Time:    Mean: %.3f Median: %.3f std: %.2f' % (np.mean(times), np.median(times), np.std(times)))

    # torch.save({
    #     'stability': validity_dict,
    #     'bond_length': all_bond_dist,
    #     'all_results': results
    # }, os.path.join(result_path, f'metrics_{args.eval_step}.pt'))
