import torch
from torch_geometric.data import Data
from rdkit import Chem

import config
from features import atom_features, edge_features

def build_graph(mol, coords, meta):
    try:
        num_atoms = mol.GetNumAtoms()
        if num_atoms < 2:
            return None, None

        x_list = []
        for atom in mol.GetAtoms():
            sym = atom.GetSymbol()
            if sym in config.ALLOWED_ELEMENTS:
                idx = config.ALLOWED_ELEMENTS.index(sym)
            else:
                idx = len(config.ALLOWED_ELEMENTS)
            x_list.append(idx)
            
        x = torch.tensor(x_list, dtype=torch.long)

        conformer = mol.GetConformer()
        pos_np = conformer.GetPositions()
        pos = torch.tensor(pos_np, dtype=torch.float32)

        data = Data(x=x, pos=pos)

        return data, meta

    except Exception as e:
        print(f"Failed to build graph for {meta.get('id', 'unknown')}: {str(e)}")
        return None, None
