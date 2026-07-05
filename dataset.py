import os
import glob
import torch
from torch.utils.data import Dataset, Subset

import config

def apply_coordinate_noise(data):
    # Sample Gaussian noise
    noise = torch.randn_like(data.pos) * config.NOISE_SCALE
    
    # Store clean positions
    data.pos_target = data.pos.clone()
    
    # Rebuild edges from noisy positions and compute distance targets
    if hasattr(data, 'edge_index') and hasattr(data, 'edge_attr'):
        row, col = data.edge_index
        
        # 1. Compute CLEAN distances
        clean_dist = torch.norm(data.pos_target[row] - data.pos_target[col], p=2, dim=-1)
        
        # 2. Perturb positions
        data.pos = data.pos + noise
        
        # 3. Compute NOISY distances
        noisy_dist = torch.norm(data.pos[row] - data.pos[col], p=2, dim=-1)
        
        # 4. Set regression target as the residual
        data.dist_target = clean_dist - noisy_dist
        
        # RBF expand the noisy distances
        centers = torch.linspace(config.RBF_MIN, config.RBF_MAX, config.RBF_NUM_CENTERS, device=noisy_dist.device)
        rbf_vals = torch.exp(-((noisy_dist.unsqueeze(1) - centers) ** 2) / (2 * (config.RBF_WIDTH ** 2)))
        
        # Overwrite edge_attr with new noisy RBFs
        data.edge_attr = rbf_vals.to(torch.float32)
    else:
        data.pos = data.pos + noise
        
    return data

class RadiusGraphTransform:
    def __call__(self, data):
        # Build radius graph dynamically using torch.cdist
        num_atoms = data.pos.size(0)
        dist_mat = torch.cdist(data.pos, data.pos)
        
        # We need device-aware identity matrix
        eye = torch.eye(num_atoms, device=data.pos.device)
        edge_index = torch.nonzero((dist_mat < config.DISTANCE_CUTOFF) & (eye == 0)).t()
        data.edge_index = edge_index
        
        row, col = edge_index
        
        # Compute exact distances
        dist = torch.norm(data.pos[row] - data.pos[col], p=2, dim=-1)
        
        # RBF expansion matching old features.py exactly
        centers = torch.linspace(config.RBF_MIN, config.RBF_MAX, config.RBF_NUM_CENTERS, device=dist.device)
        rbf_vals = torch.exp(-((dist.unsqueeze(1) - centers) ** 2) / (2 * (config.RBF_WIDTH ** 2)))
        
        data.edge_attr = rbf_vals.to(torch.float32)
        return data

class CoordinateNoiseTransform:
    def __call__(self, data):
        return apply_coordinate_noise(data)

class MoleculeGraphDataset(Dataset):
    def __init__(self, root_dir=config.PROCESSED_DIR, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.files = sorted(
            [f for f in glob.glob(os.path.join(root_dir, "*.pt"))
             if f.endswith(".pdb.pt") or f.endswith(".sdf.pt")]
        )
        
    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, idx):
        filepath = self.files[idx]
        data = torch.load(filepath, weights_only=False)
        
        # Apply mandatory RadiusGraphTransform
        rg_transform = RadiusGraphTransform()
        data = rg_transform(data)
        
        if self.transform is not None:
            data = self.transform(data)
            
        return data

def make_splits(dataset, val_split=config.VAL_SPLIT, seed=config.RANDOM_SEED):
    num_data = len(dataset)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(num_data, generator=generator).tolist()
    
    split_idx = int(num_data * (1 - val_split))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    return train_dataset, val_dataset
