import os
import sys
import numpy as np

# Mock freesasa
from unittest.mock import MagicMock
sys.modules['freesasa'] = MagicMock()

from atom3d.datasets import LMDBDataset

# Remove the downloaded archive to free up space
archive_path = "data/atom3d_res/res.tar.gz"
if os.path.exists(archive_path):
    print("Deleting compressed archive to free up disk space...")
    os.remove(archive_path)

splits = {
    'test': 'data/atom3d_res/split-by-cath-topology/data/test',
    'val': 'data/atom3d_res/split-by-cath-topology/data/val',
    'train': 'data/atom3d_res/split-by-cath-topology/data/train'
}

for name, path in splits.items():
    if not os.path.exists(path):
        print(f"Path {path} does not exist.")
        continue
    
    dataset = LMDBDataset(path)
    total_structures = len(dataset)
    
    # We sample a representative set of structures to be super fast
    sample_size = min(total_structures, 300)
    print(f"\nAnalyzing split: {name} ({total_structures} total structures, sampling {sample_size})...")
    
    atom_counts = []
    for i in range(sample_size):
        item = dataset[i]
        if 'subunit_indices' in item:
            for idx_list in item['subunit_indices']:
                atom_counts.append(len(idx_list))
        elif 'atoms' in item:
            atom_counts.append(len(item['atoms']))
            
    arr = np.array(atom_counts)
    print(f"  Estimated total environments in split: {len(arr) * (total_structures / sample_size):.0f}")
    print(f"  Atoms per environment:")
    print(f"    Min:    {arr.min()}")
    print(f"    Max:    {arr.max()}")
    print(f"    Mean:   {arr.mean():.1f}")
    print(f"    Median: {np.median(arr):.1f}")
    print(f"    Std:    {arr.std():.1f}")
