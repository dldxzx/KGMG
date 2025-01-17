import os
import re
import torch
from pathlib import Path
import argparse

import pandas as pd
from rdkit import Chem
from tqdm import tqdm
import time


def calculate_smina_score(pdb_file, sdf_file):
    # add '-o <name>_smina.sdf' if you want to see the output
    out = os.popen(f'smina.static -l {sdf_file} -r {pdb_file} '
                   f'--score_only').read()
    matches = re.findall(
        r"Affinity:[ ]+([+-]?[0-9]*[.]?[0-9]+)[ ]+\(kcal/mol\)", out)
    return [float(x) for x in matches]


def sdf_to_pdbqt(sdf_file, pdbqt_outfile, mol_id):
    os.popen(f'/path/bin/obabel {sdf_file} -O {pdbqt_outfile} ').read()
    return pdbqt_outfile

def calculate_qvina2_score(receptor_file, sdf_file, out_dir, size=20,
                           exhaustiveness=16, return_rdmol=False, index=None):
    receptor_file = Path(receptor_file)
    receptor_name = os.path.basename(receptor_file)[:4]
    

    receptor_pdbqt_file = receptor_file
    scores = []
    rdmols = []  # for if return rdmols
    pdb_flag = False
    if type(sdf_file) == str:
        sdf_file = Path(sdf_file)
        suppl = Chem.SDMolSupplier(str(sdf_file), sanitize=False)
        pdb_flag = True
    else:
        suppl = [sdf_file]
        ligand_name = f'{receptor_name}_{index}'
        os.makedirs(os.path.join(out_dir,receptor_name), exist_ok=True)
        ligand_file = os.path.join(out_dir,receptor_name, ligand_name + '.sdf')
        if not Path(ligand_file).exists() or Path(ligand_file).stat().st_size== 0:
            sdf_writer = Chem.SDWriter(ligand_file)
            sdf_writer.write(sdf_file)
            sdf_writer.close()
        sdf_file = ligand_file
    for i, mol in enumerate(suppl):  # sdf file may contain several ligands
        if index is not None:
            i = index
        # print(f'sample_idx:{sample_idx}, i:{i}, smiles:{Chem.MolToSmiles(mol)} receptor_pdbqt_file{receptor_pdbqt_file}')
        ligand_name = f'{receptor_name}_{i}'
        ligand_name = os.path.basename(sdf_file)
        ligand_name = os.path.basename(sdf_file).split('.sdf')[0]
        ligand_name = f'{ligand_name}_{i}'
        # prepare ligand
        if pdb_flag:
            ligand_pdbqt_file = Path(out_dir, ligand_name + '.pdbqt')
            out_sdf_file = Path(out_dir, ligand_name + '_out.sdf')
            out_pdbqt_file = Path(out_dir, ligand_name + '_out.pdbqt')
        else:
            ligand_pdbqt_file = Path(out_dir, receptor_name, ligand_name + '.pdbqt')
            out_sdf_file = Path(out_dir, receptor_name, ligand_name + '_out.sdf')
            out_pdbqt_file = Path(out_dir, receptor_name, ligand_name + '_out.pdbqt')

        # out_pdbqt_file = Path(out_dir, ligand_name + '_out.pdbqt')
        # you have to assdign your own mol envrionment
        if out_pdbqt_file.exists() and not out_sdf_file.exists():
            os.popen(f'/path/bin/obabel {out_pdbqt_file} -O {out_sdf_file}').read()
        if out_sdf_file.exists() and out_sdf_file.stat().st_size != 0:
            # print(out_sdf_file)
            with open(out_sdf_file, 'r') as f:
                scores.append(
                    min([float(x.split()[2]) for x in f.readlines()
                         if x.startswith(' VINA RESULT:')])
                )
        else:
            sdf_to_pdbqt(sdf_file, ligand_pdbqt_file, i)
            # center box at ligand's center of mass
            cx, cy, cz = mol.GetConformer().GetPositions().mean(0)
            # run QuickVina 2
            out = os.popen(
                f'/home/user/fpk/KGMG/qvina2.1 --receptor {receptor_pdbqt_file} '
                f'--ligand {ligand_pdbqt_file} '
                f'--center_x {cx:.4f} --center_y {cy:.4f} --center_z {cz:.4f} '
                f'--size_x {size} --size_y {size} --size_z {size} '
                f'--exhaustiveness {exhaustiveness}'
            ).read()
            out_split = out.splitlines()
            # print(f'out_split:{out_split}')
            best_idx = out_split.index('-----+------------+----------+----------') + 1
            best_line = out_split[best_idx].split()
            assert best_line[0] == '1'
            scores.append(float(best_line[1]))

            
            if out_pdbqt_file.exists():
                os.popen(f'/path/bin/obabel {out_pdbqt_file} -O {out_sdf_file}').read()

        if return_rdmol:
            rdmol = Chem.SDMolSupplier(str(out_sdf_file))[0]
            rdmols.append(rdmol)

    if return_rdmol:
        return scores, rdmols
    else:
        return scores
