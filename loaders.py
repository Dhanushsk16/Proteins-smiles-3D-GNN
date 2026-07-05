import os
import numpy as np
from rdkit import Chem

def standardize(mol):
    if mol is None:
        return None
    try:
        mol = Chem.RemoveHs(mol)
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None

def load_pdb(filepath):
    try:
        mol = Chem.MolFromPDBFile(filepath, removeHs=False, sanitize=False)
        if mol is None:
            return None
        
        mol = standardize(mol)
        if mol is None:
            return None
            
        conformer = mol.GetConformer()
        coords = conformer.GetPositions()
        
        plddt_values = []
        for atom in mol.GetAtoms():
            info = atom.GetPDBResidueInfo()
            if info is not None:
                plddt_values.append(info.GetTempFactor())
                
        plddt_mean = float(np.mean(plddt_values)) if plddt_values else 0.0
        
        meta = {
            'source': 'pdb',
            'id': os.path.basename(filepath),
            'plddt_mean': plddt_mean
        }
        
        return (mol, coords, meta)
    except Exception:
        return None

def load_sdf(filepath):
    results = []
    try:
        suppl = Chem.SDMolSupplier(filepath, removeHs=False, sanitize=False)
        for mol in suppl:
            if mol is None:
                continue
                
            mol = standardize(mol)
            if mol is None:
                continue
                
            conformer = mol.GetConformer()
            coords = conformer.GetPositions()
            
            meta = {
                'source': 'sdf',
                'id': os.path.basename(filepath)
            }
            
            results.append((mol, coords, meta))
        return results
    except Exception:
        return []

def load_any(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.pdb':
        return load_pdb(filepath)
    elif ext == '.sdf':
        return load_sdf(filepath)
    else:
        return None
