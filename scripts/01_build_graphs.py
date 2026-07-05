import sys
import os
import glob
import logging
import json
from tqdm import tqdm
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from loaders import load_any
from filters import apply_all_filters
from graph_builder import build_graph

def setup_logger():
    logger = logging.getLogger("build_graphs")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(config.LOG_DIR, "build_graphs.log"))
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

def main():
    logger = setup_logger()
    
    pdb_files = sorted(glob.glob(os.path.join(config.PDB_DIR, "*.pdb")))
    sdf_files = sorted(glob.glob(os.path.join(config.SDF_DIR, "*.sdf")))
    
    if config.DEBUG_MODE:
        logger.info(f"DEBUG_MODE is True. Taking subset of size {config.SUBSET_SIZE}")
        half = config.SUBSET_SIZE // 2
        all_files = sdf_files[:half] + pdb_files[:(config.SUBSET_SIZE - half)]
    else:
        all_files = pdb_files + sdf_files
        
    logger.info(f"Starting graph build over {len(all_files)} files")
    
    counts = {
        'files_seen': len(all_files),
        'molecules_loaded': 0,
        'graph_failed': 0,
        'load_error': 0,
        'graphs_saved': 0
    }
    rejection_reasons = {}
    metadata_index = {}
    
    for filepath in tqdm(all_files, desc="Building graphs"):
        try:
            results = load_any(filepath)
            if not results:
                continue
                
            if not isinstance(results, list):
                results = [results]
                
            for mol, coords, meta in results:
                counts['molecules_loaded'] += 1
                
                keep, reasons = apply_all_filters(mol, coords, meta)
                if not keep:
                    for r in reasons:
                        rejection_reasons[r] = rejection_reasons.get(r, 0) + 1
                    continue
                    
                data, meta_out = build_graph(mol, coords, meta)
                if data is None:
                    counts['graph_failed'] += 1
                    continue
                    
                safe_id = meta.get('id', 'unknown').replace('/', '_').replace('\\', '_')
                save_path = os.path.join(config.PROCESSED_DIR, f"{safe_id}.pt")
                torch.save(data, save_path)
                
                metadata_index[safe_id] = meta_out
                
                counts['graphs_saved'] += 1
                
        except Exception as e:
            logger.error(f"Error processing {os.path.basename(filepath)}: {e}")
            counts['load_error'] += 1

    print("\n--- SUMMARY ---")
    print(f"Files seen: {counts['files_seen']}")
    print(f"Molecules loaded: {counts['molecules_loaded']}")
    
    print("\nRejections per reason:")
    if rejection_reasons:
        for r, count in rejection_reasons.items():
            print(f"  {r}: {count}")
    else:
        print("  None")
        
    print(f"\nGraph builder failed: {counts['graph_failed']}")
    print(f"Load/parse errors: {counts['load_error']}")
    print(f"Total graphs saved: {counts['graphs_saved']}")
    
    logger.info(f"Run complete. Saved {counts['graphs_saved']} graphs.")
    
    index_path = os.path.join(config.PROCESSED_DIR, "metadata_index.json")
    with open(index_path, 'w') as f:
        json.dump(metadata_index, f, indent=2)
    logger.info(f"Saved metadata index to {index_path}")

if __name__ == "__main__":
    main()
