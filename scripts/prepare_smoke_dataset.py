import os
import glob
import random
import torch
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def main():
    processed_dir = config.PROCESSED_DIR
    pt_files = sorted(glob.glob(os.path.join(processed_dir, "*.pt")))
    
    pdb_files = []
    sdf_files = []
    
    for f in pt_files:
        if f.endswith(".pdb.pt"):
            pdb_files.append(f)
        elif f.endswith(".sdf.pt"):
            sdf_files.append(f)
        else:
            try:
                data = torch.load(f, weights_only=False)
                src = getattr(data, "source", None)
                if src in ["alphafold", "pdb"]:
                    pdb_files.append(f)
                elif src in ["sdf", "pubchem"]:
                    sdf_files.append(f)
            except Exception:
                pass

    pdb_files = sorted(list(set(pdb_files)))
    sdf_files = sorted(list(set(sdf_files)))

    print(f"Total found PDB files: {len(pdb_files)}")
    print(f"Total found SDF files: {len(sdf_files)}")

    rng = random.Random(config.RANDOM_SEED)
    sampled_pdb = rng.sample(pdb_files, 100)
    sampled_sdf = rng.sample(sdf_files, 100)

    sampled_all = sampled_pdb + sampled_sdf
    rng.shuffle(sampled_all)

    train_files = sampled_all[:140]
    val_files = sampled_all[140:170]
    test_files = sampled_all[170:]

    split_data = {
        "train": train_files,
        "val": val_files,
        "test": test_files
    }

    save_path = os.path.join(processed_dir, "smoke_split.pt")
    torch.save(split_data, save_path)
    print(f"Saved smoke split with {len(train_files)} train, {len(val_files)} val, {len(test_files)} test files to {save_path}")

if __name__ == "__main__":
    main()
