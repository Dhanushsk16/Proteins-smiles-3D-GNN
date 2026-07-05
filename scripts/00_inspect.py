import sys
import os
import glob
from collections import Counter
from rdkit import Chem

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def get_pdb_stats(mol):
    if mol is None:
        return None
    heavy_atoms = mol.GetNumHeavyAtoms()
    elements = set(atom.GetSymbol() for atom in mol.GetAtoms())
    
    residues = set()
    plddt_values = []
    
    for atom in mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info is not None:
            residues.add((info.GetChainId(), info.GetResidueNumber()))
            plddt_values.append(info.GetTempFactor())
            
    seq_len = len(residues)
    avg_plddt = sum(plddt_values) / len(plddt_values) if plddt_values else 0.0
    
    has_coords = mol.GetNumConformers() > 0
    is_single = len(Chem.GetMolFrags(mol)) == 1
    
    return {
        'heavy_atoms': heavy_atoms,
        'seq_len': seq_len,
        'avg_plddt': avg_plddt,
        'elements': elements,
        'has_coords': has_coords,
        'is_single': is_single
    }

def get_sdf_stats(mol):
    if mol is None:
        return None
    heavy_atoms = mol.GetNumHeavyAtoms()
    elements = set(atom.GetSymbol() for atom in mol.GetAtoms())
    has_coords = mol.GetNumConformers() > 0
    is_single = len(Chem.GetMolFrags(mol)) == 1
    
    return {
        'heavy_atoms': heavy_atoms,
        'seq_len': 0,
        'avg_plddt': 0.0,
        'elements': elements,
        'has_coords': has_coords,
        'is_single': is_single
    }

def main():
    pdb_files = glob.glob(os.path.join(config.PDB_DIR, "*.pdb"))
    sdf_files = glob.glob(os.path.join(config.SDF_DIR, "*.sdf"))
    
    print(f"Found {len(pdb_files)} PDB files and {len(sdf_files)} SDF files.")
    
    print("\n--- SAMPLE FILES ---")
    for file in pdb_files[:5]:
        mol = Chem.MolFromPDBFile(file)
        stats = get_pdb_stats(mol)
        print(f"PDB {os.path.basename(file)}: {stats}")
        
    for file in sdf_files[:5]:
        suppl = Chem.SDMolSupplier(file)
        if len(suppl) > 0:
            mol = suppl[0]
            stats = get_sdf_stats(mol)
            print(f"SDF {os.path.basename(file)}: {stats}")

    print("\n--- AGGREGATE STATISTICS ---")
    
    all_heavy_atoms = []
    all_seq_lens = []
    all_plddts = []
    all_elements = set()
    
    parse_failures = 0
    missing_coords = 0
    multi_component = 0
    
    for file in pdb_files:
        mol = Chem.MolFromPDBFile(file)
        if mol is None:
            parse_failures += 1
            print(f"WARNING: Parse failure on {os.path.basename(file)}")
            continue
            
        stats = get_pdb_stats(mol)
        all_heavy_atoms.append(stats['heavy_atoms'])
        all_seq_lens.append(stats['seq_len'])
        if stats['avg_plddt'] > 0:
            all_plddts.append(stats['avg_plddt'])
        all_elements.update(stats['elements'])
        
        if not stats['has_coords']:
            missing_coords += 1
            print(f"WARNING: Missing coords in {os.path.basename(file)}")
        if not stats['is_single']:
            multi_component += 1
            print(f"WARNING: Multi-component in {os.path.basename(file)}")

    for file in sdf_files:
        suppl = Chem.SDMolSupplier(file)
        if len(suppl) == 0:
            parse_failures += 1
            print(f"WARNING: Parse failure on {os.path.basename(file)}")
            continue
        
        for mol in suppl:
            if mol is None:
                parse_failures += 1
                continue
                
            stats = get_sdf_stats(mol)
            all_heavy_atoms.append(stats['heavy_atoms'])
            all_elements.update(stats['elements'])
            
            if not stats['has_coords']:
                missing_coords += 1
                print(f"WARNING: Missing coords in {os.path.basename(file)}")
            if not stats['is_single']:
                multi_component += 1
                print(f"WARNING: Multi-component in {os.path.basename(file)}")

    if all_heavy_atoms:
        print(f"Heavy atoms: min={min(all_heavy_atoms)}, max={max(all_heavy_atoms)}, avg={sum(all_heavy_atoms)/len(all_heavy_atoms):.1f}")
    if all_seq_lens:
        print(f"Seq lengths (PDB): min={min(all_seq_lens)}, max={max(all_seq_lens)}, avg={sum(all_seq_lens)/len(all_seq_lens):.1f}")
    if all_plddts:
        print(f"pLDDT (PDB): min={min(all_plddts):.1f}, max={max(all_plddts):.1f}, avg={sum(all_plddts)/len(all_plddts):.1f}")
        
    print(f"All elements seen: {all_elements}")
    print(f"Parse failures: {parse_failures}")
    print(f"Missing coords: {missing_coords}")
    print(f"Multi-component: {multi_component}")
    
    print("\n--- ANOMALIES AGAINST ASSUMPTIONS ---")
    unexpected_elements = all_elements - set(config.ALLOWED_ELEMENTS)
    if unexpected_elements:
        print(f"WARNING: Found elements outside ALLOWED_ELEMENTS: {unexpected_elements}")
    else:
        print("OK: Elements match ALLOWED_ELEMENTS.")
        
    if all_heavy_atoms and max(all_heavy_atoms) > config.MAX_HEAVY_ATOMS:
        print(f"WARNING: Found molecules larger than MAX_HEAVY_ATOMS ({config.MAX_HEAVY_ATOMS})")
    else:
        print(f"OK: No molecules larger than MAX_HEAVY_ATOMS ({config.MAX_HEAVY_ATOMS})")
        
    if all_heavy_atoms and min(all_heavy_atoms) < config.MIN_HEAVY_ATOMS:
        print(f"WARNING: Found molecules smaller than MIN_HEAVY_ATOMS ({config.MIN_HEAVY_ATOMS})")
    else:
        print(f"OK: No molecules smaller than MIN_HEAVY_ATOMS ({config.MIN_HEAVY_ATOMS})")
    
    if all_plddts and min(all_plddts) < config.PLDDT_CUTOFF:
        print(f"WARNING: Found pLDDT below PLDDT_CUTOFF ({config.PLDDT_CUTOFF})")
    else:
        print(f"OK: No pLDDT below PLDDT_CUTOFF ({config.PLDDT_CUTOFF})")
        
    if all_seq_lens and max(all_seq_lens) > config.MAX_PEPTIDE_LENGTH:
        print(f"WARNING: Found sequences longer than MAX_PEPTIDE_LENGTH ({config.MAX_PEPTIDE_LENGTH})")
    else:
        print(f"OK: No sequences longer than MAX_PEPTIDE_LENGTH ({config.MAX_PEPTIDE_LENGTH})")

if __name__ == "__main__":
    main()
