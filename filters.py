import numpy as np
from rdkit import Chem
import config

"""
Filtered at download time, intentionally not re-checked here:
- max heavy atoms
- max peptide length
- pLDDT cutoff
- single-component
"""

def passes_element_filter(mol):
    allowed = set(config.ALLOWED_ELEMENTS)
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in allowed:
            return False
    return True

def passes_size_filter(mol):
    return mol.GetNumHeavyAtoms() >= config.MIN_HEAVY_ATOMS

def passes_coordinates_check(mol, coords):
    if mol.GetNumConformers() == 0:
        return False
    if coords is None:
        return False
    if len(coords) != mol.GetNumAtoms():
        return False
    
    if len(coords) > 0:
        if np.all(coords == 0.0):
            return False
        stds = np.std(coords, axis=0)
        if np.all(stds < 1e-4):
            return False
            
    return True

def passes_sanitization(mol):
    try:
        res = Chem.SanitizeMol(mol, catchErrors=True)
        return res == Chem.SanitizeFlags.SANITIZE_NONE
    except Exception:
        return False

def apply_all_filters(mol, coords, meta):
    reasons = []
    
    if not passes_element_filter(mol):
        reasons.append('element')
        
    if not passes_size_filter(mol):
        reasons.append('size')
        
    if not passes_coordinates_check(mol, coords):
        reasons.append('coordinates')
        
    if not passes_sanitization(mol):
        reasons.append('sanitization')
        
    keep = len(reasons) == 0
    return keep, reasons
