
from rdkit import Chem, DataStructs
def calculate_diversity(pocket_mols):
    if len(pocket_mols) < 2:
        return 0.0
    div = 0
    total = 0
    for i in range(len(pocket_mols)):
        for j in range(i + 1, len(pocket_mols)):
            div += 1 - similarity(pocket_mols[i], pocket_mols[j])
            total += 1
    return div / total

def similarity(mol_a, mol_b):
    # fp1 = AllChem.GetMorganFingerprintAsBitVect(
    #     mol_a, 2, nBits=2048, useChirality=False)
    # fp2 = AllChem.GetMorganFingerprintAsBitVect(
    #     mol_b, 2, nBits=2048, useChirality=False)
    fp1 = Chem.RDKFingerprint(mol_a)
    fp2 = Chem.RDKFingerprint(mol_b)
    return DataStructs.TanimotoSimilarity(fp1, fp2)