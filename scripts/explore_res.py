"""
Explore the ATOM3D RES (Residue Identity) dataset.
Reports: number of instances (residue environments), avg/min/max atoms per environment.
"""
import os
import sys
import time
import urllib.request
import tarfile
import numpy as np

# Mock freesasa since it can't be compiled on Windows without MSVC
from unittest.mock import MagicMock
sys.modules['freesasa'] = MagicMock()

from atom3d.datasets import LMDBDataset

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "atom3d_res")
ZENODO_URL = "https://zenodo.org/record/5026743/files/RES-split-by-cath-topology.tar.gz?download=1"
TAR_PATH = os.path.join(DATA_DIR, "res.tar.gz")


def download_with_progress(url, dest):
    """Download a file with a progress indicator."""
    import requests as req_lib
    print(f"Downloading RES dataset (~4.1 GB compressed)...")
    print(f"  URL:  {url}")
    print(f"  Dest: {dest}")
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    resp = req_lib.get(url, stream=True, headers={"User-Agent": "python-requests/2.32"})
    resp.raise_for_status()
    total = int(resp.headers.get("Content-Length", 0))
    downloaded = 0
    block = 1024 * 1024  # 1 MB
    t0 = time.time()

    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=block):
            f.write(chunk)
            downloaded += len(chunk)
            pct = downloaded / total * 100 if total else 0
            elapsed = time.time() - t0
            speed = downloaded / elapsed / 1e6 if elapsed > 0 else 0
            print(f"\r  {downloaded / 1e9:.2f} / {total / 1e9:.2f} GB  ({pct:.1f}%)  {speed:.1f} MB/s", end="", flush=True)
    print()



def extract_tar(tar_path, dest_dir):
    """Extract a .tar.gz archive."""
    print(f"Extracting {tar_path} ...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(dest_dir)
    print("  Done.")


def find_lmdb_dirs(root):
    """Find all LMDB directories (contain data.mdb) under root."""
    dirs = []
    for dirpath, dirnames, filenames in os.walk(root):
        if "data.mdb" in filenames:
            dirs.append(dirpath)
    return sorted(dirs)


def compute_stats(lmdb_path, split_name, max_items=None):
    """
    Load an LMDB dataset and compute per-environment atom counts.

    Each LMDB item is one PDB structure containing multiple residue
    environments. The item has:
      - 'atoms': DataFrame of all atoms
      - 'labels': DataFrame with one row per environment
      - 'subunit_indices': list-of-lists mapping each environment to rows in 'atoms'

    We count atoms per environment (= length of each subunit_indices entry).
    """
    dataset = LMDBDataset(lmdb_path)
    n_structures = len(dataset)
    print(f"\n--- {split_name} split ---")
    print(f"  LMDB entries (PDB structures): {n_structures}")

    atom_counts = []
    total_envs = 0
    limit = max_items if max_items else n_structures

    for i in range(min(limit, n_structures)):
        item = dataset[i]

        # Each item has 'subunit_indices' listing atom rows per environment
        if 'subunit_indices' in item:
            for idx_list in item['subunit_indices']:
                atom_counts.append(len(idx_list))
                total_envs += 1
        elif 'labels' in item:
            # fallback: count labels as environments, atoms as total
            n_env = len(item['labels'])
            n_atoms = len(item['atoms'])
            avg_per_env = n_atoms / n_env if n_env > 0 else n_atoms
            for _ in range(n_env):
                atom_counts.append(int(avg_per_env))
                total_envs += 1
        else:
            # single-environment item
            atom_counts.append(len(item['atoms']))
            total_envs += 1

        if (i + 1) % 500 == 0 or i == min(limit, n_structures) - 1:
            print(f"  Processed {i + 1}/{min(limit, n_structures)} structures, {total_envs} environments so far...", flush=True)

    arr = np.array(atom_counts)
    sampled_note = "" if limit >= n_structures else f" (sampled from {limit}/{n_structures} structures)"
    print(f"\n  Total environments{sampled_note}: {len(arr)}")
    print(f"  Atoms per environment:")
    print(f"    Min:    {arr.min()}")
    print(f"    Max:    {arr.max()}")
    print(f"    Mean:   {arr.mean():.1f}")
    print(f"    Median: {np.median(arr):.1f}")
    print(f"    Std:    {arr.std():.1f}")
    return arr


def main():
    # ---- Step 1: Download if needed ----
    lmdb_dirs = find_lmdb_dirs(DATA_DIR)
    if not lmdb_dirs:
        if not os.path.exists(TAR_PATH):
            download_with_progress(ZENODO_URL, TAR_PATH)
        extract_tar(TAR_PATH, DATA_DIR)
        lmdb_dirs = find_lmdb_dirs(DATA_DIR)

    if not lmdb_dirs:
        print("ERROR: No LMDB directories found after extraction.")
        sys.exit(1)

    print(f"\nFound {len(lmdb_dirs)} LMDB split(s):")
    for d in lmdb_dirs:
        print(f"  {d}")

    # ---- Step 2: Compute stats per split ----
    all_counts = []
    for lmdb_dir in lmdb_dirs:
        split_name = os.path.basename(lmdb_dir)
        counts = compute_stats(lmdb_dir, split_name)
        all_counts.append(counts)

    # ---- Step 3: Overall stats ----
    combined = np.concatenate(all_counts)
    print(f"\n=== OVERALL RES DATASET ===")
    print(f"  Total environments: {len(combined)}")
    print(f"  Atoms per environment:")
    print(f"    Min:    {combined.min()}")
    print(f"    Max:    {combined.max()}")
    print(f"    Mean:   {combined.mean():.1f}")
    print(f"    Median: {np.median(combined):.1f}")
    print(f"    Std:    {combined.std():.1f}")


if __name__ == "__main__":
    main()
