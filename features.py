import numpy as np
from rdkit import Chem
import config

DEGREE_CHOICES = list(range(11))
NUM_HS_CHOICES = list(range(11))
CHARGE_CHOICES = [-2, -1, 0, 1, 2]
HYBRIDIZATION_CHOICES = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
CHIRALITY_CHOICES = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER,
]
BOND_TYPE_CHOICES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
    None,
]

RBF_CENTERS = np.linspace(config.RBF_MIN, config.RBF_MAX, config.RBF_NUM_CENTERS)

def onehot(value, choices):
    encoding = [0] * (len(choices) + 1)
    if value in choices:
        encoding[choices.index(value)] = 1
    else:
        encoding[-1] = 1
    return encoding

def atom_features(atom):
    features = []
    
    features.extend(onehot(atom.GetSymbol(), config.ALLOWED_ELEMENTS))
    features.extend(onehot(atom.GetDegree(), DEGREE_CHOICES))
    features.extend(onehot(atom.GetTotalNumHs(), NUM_HS_CHOICES))
    features.extend(onehot(atom.GetFormalCharge(), CHARGE_CHOICES))
    features.extend(onehot(atom.GetHybridization(), HYBRIDIZATION_CHOICES))
    features.extend(onehot(atom.GetChiralTag(), CHIRALITY_CHOICES))
    
    features.append(1 if atom.GetIsAromatic() else 0)
    features.append(1 if atom.IsInRing() else 0)
    
    return features

def rbf_expand(distance, centers, width):
    return np.exp(-((distance - centers) ** 2) / (2 * (width ** 2)))

def bond_type_onehot(bond_or_none):
    return onehot(bond_or_none, BOND_TYPE_CHOICES)

def edge_features(distance, bond_or_none):
    rbf_vals = rbf_expand(distance, RBF_CENTERS, config.RBF_WIDTH).tolist()
    bond_vals = bond_type_onehot(bond_or_none)
    return rbf_vals + bond_vals

_dummy_mol = Chem.MolFromSmiles('C')
_dummy_atom = _dummy_mol.GetAtomWithIdx(0)
NODE_FEATURE_DIM = len(atom_features(_dummy_atom))
EDGE_FEATURE_DIM = len(edge_features(1.0, None))
