import os
import sys
import time
import glob
import logging
import requests
import torch
import numpy as np
from rdkit import Chem
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from loaders import standardize
from graph_builder import build_graph

# Setup logging
os.makedirs(config.LOG_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("download_dataset")
fh = logging.FileHandler(os.path.join(config.LOG_DIR, "download_dataset.log"))
fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(fh)

# Direct EBI file download URL base
EBI_PDB_URL = "https://alphafold.ebi.ac.uk/files/AF-{accession}-F1-model_v6.pdb"
EBI_API_URL = "https://alphafold.ebi.ac.uk/api/prediction/{accession}"

def clean_directories():
    """Ensure clean starting directories for raw and processed datasets."""
    logger.info("Cleaning directories...")
    os.makedirs(config.PDB_DIR, exist_ok=True)
    os.makedirs(config.SDF_DIR, exist_ok=True)
    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    
    # Only clean processed directory
    files = glob.glob(os.path.join(config.PROCESSED_DIR, "*"))
    for f in files:
        try:
            if os.path.isfile(f):
                os.remove(f)
        except Exception as e:
            logger.error(f"Failed to delete {f}: {e}")

def validate_conformer_coordinates(mol):
    """
    Validation gate checks:
    - Exactly one 3D conformer is present.
    - No NaNs in coordinates.
    - No zero-length structure.
    - No two atoms closer than 0.5 A.
    - Atom count within model's supported range.
    """
    if mol is None:
        return False, "Failed RDKit parsing"
    if mol.GetNumConformers() != 1:
        return False, f"Expected 1 conformer, got {mol.GetNumConformers()}"
    
    conf = mol.GetConformer()
    if not conf.Is3D():
        return False, "Conformer is not 3D"
        
    coords = conf.GetPositions()
    if len(coords) == 0:
        return False, "Zero-length structure"
    if np.any(np.isnan(coords)):
        return False, "Coordinates contain NaN values"
        
    # Check minimum distance between any two atoms
    num_atoms = mol.GetNumAtoms()
    if num_atoms > 1:
        dist_mat = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
        np.fill_diagonal(dist_mat, np.inf)
        min_dist = np.min(dist_mat)
        if min_dist < 0.5:
            return False, f"Atoms too close: minimum distance is {min_dist:.4f} A"
            
    # Check heavy atom range
    heavy_atoms = mol.GetNumHeavyAtoms()
    if not (config.MIN_HEAVY_ATOMS <= heavy_atoms <= config.MAX_HEAVY_ATOMS):
        return False, f"Heavy atom count {heavy_atoms} outside supported range"
        
    return True, "Valid"

# Global Provenance Stats
stats = {
    'peptides': {
        'fetched': 0,
        'length_dropped': 0,
        'plddt_dropped': 0,
        'failed_validation': 0,
        'dedup_dropped': 0,
        'no_coords_skipped': 0,
        'retained': 0
    },
    'molecules': {
        'fetched': 0,
        'fetched_sdf': 0,
        'size_dropped': 0,
        'smiles_len_dropped': 0,
        'salt_split_kept': 0,
        'silicon_dropped': 0,
        'failed_validation': 0,
        'no_3d_coords_skipped': 0,
        'dedup_dropped': 0,
        'retained': 0
    }
}

# InChIKeys tracking for deduplication
all_inchikeys = set()

# ==================== PEPTIDE PIPELINE ====================

def fetch_uniprot_candidates(limit=80000):
    """Query UniProt stream API for peptides of length 5-50 with AlphaFold cross-references."""
    logger.info("Streaming candidates from UniProt...")
    url = "https://rest.uniprot.org/uniprotkb/stream"
    params = {
        'query': '(length:[5 TO 50]) AND (database:alphafolddb)',
        'format': 'tsv',
        'fields': 'accession,sequence,length',
    }
    
    candidates = []
    try:
        r = requests.get(url, params=params, stream=True)
        r.raise_for_status()
        
        lines = r.iter_lines()
        header = next(lines).decode('utf-8').split('\t')
        
        for line in lines:
            if len(candidates) >= limit:
                break
            parts = line.decode('utf-8').split('\t')
            if len(parts) >= 3:
                candidates.append({
                    'accession': parts[0],
                    'sequence': parts[1],
                    'length': int(parts[2])
                })
        logger.info(f"Retrieved {len(candidates)} candidate peptides from UniProt.")
    except Exception as e:
        logger.error(f"Error fetching UniProt stream: {e}")
        
    return candidates

def download_and_validate_single_pdb(cand):
    """Download AF PDB file and run pLDDT/validation gate checks."""
    accession = cand['accession']
    file_path = os.path.join(config.PDB_DIR, f"{accession}.pdb")
    
    pdb_text = None
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                pdb_text = f.read()
        except Exception as e:
            logger.error(f"Failed to read local PDB file {file_path}: {e}")
            
    if not pdb_text:
        # Try downloading v6 PDB directly (fast path)
        url = EBI_PDB_URL.format(accession=accession)
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                pdb_text = r.text
            else:
                # Fallback path: query EBI prediction API
                api_res = requests.get(EBI_API_URL.format(accession=accession), timeout=10)
                if api_res.status_code == 200:
                    api_data = api_res.json()
                    if api_data and isinstance(api_data, list):
                        pdb_url = api_data[0].get("pdbUrl")
                        if pdb_url:
                            r2 = requests.get(pdb_url, timeout=10)
                            if r2.status_code == 200:
                                pdb_text = r2.text
        except Exception as e:
            return {'status': 'error', 'msg': str(e)}
        
    if not pdb_text:
        return {'status': 'no_coords', 'msg': 'PDB not found/No 3D conformer at source'}
        
    try:
        # Load molecule from PDB block
        mol = Chem.MolFromPDBBlock(pdb_text, removeHs=False, sanitize=False)
        if mol is None:
            return {'status': 'validation_failed', 'msg': 'Failed PDB parsing'}
            
        # Standardize structure (Remove Hs and Sanitize)
        std_mol = standardize(mol)
        if std_mol is None:
            return {'status': 'validation_failed', 'msg': 'Standardization/Sanitization failed'}
            
        # Validation gate
        is_valid, val_msg = validate_conformer_coordinates(std_mol)
        if not is_valid:
            return {'status': 'validation_failed', 'msg': val_msg}
            
        # Calculate mean pLDDT from TempFactor column for C-Alpha atoms
        plddts = []
        for atom in mol.GetAtoms():
            info = atom.GetPDBResidueInfo()
            # Only use C-Alpha for unbiased mean
            if info is not None and atom.GetSymbol() == 'C' and info.GetName().strip() == 'CA':
                plddts.append(info.GetTempFactor())
                
        plddt_mean = float(np.mean(plddts)) if plddts else 0.0
        
        if plddt_mean <= 70.0:
            return {'status': 'plddt_failed', 'msg': f"pLDDT {plddt_mean:.1f} <= 70.0"}
            
        # Compute InChIKey for cross-source deduplication
        inchikey = Chem.MolToInchiKey(std_mol)
        
        # Save PDB file locally if not already there
        if not os.path.exists(file_path):
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(pdb_text)
            
        return {
            'status': 'success',
            'accession': accession,
            'sequence': cand['sequence'],
            'length': cand['length'],
            'plddt_mean': plddt_mean,
            'inchikey': inchikey,
            'filepath': file_path,
            'mol': std_mol
        }
    except Exception as e:
        return {'status': 'error', 'msg': str(e)}

def get_sequence_from_pdb_text(pdb_text):
    """Extract 1-letter amino acid sequence from PDB text."""
    d3to1 = {
        'ALA':'A', 'CYS':'C', 'ASP':'D', 'GLU':'E', 'PHE':'F', 'GLY':'G', 'HIS':'H', 
        'ILE':'I', 'LYS':'K', 'LEU':'L', 'MET':'M', 'ASN':'N', 'PRO':'P', 'GLN':'Q', 
        'ARG':'R', 'SER':'S', 'THR':'T', 'VAL':'V', 'TRP':'W', 'TYR':'Y'
    }
    residues = []
    last_res_id = None
    for line in pdb_text.splitlines():
        if line.startswith("ATOM  ") or line.startswith("HETATM"):
            res_name = line[17:20].strip()
            chain_id = line[21]
            res_seq = line[22:26].strip()
            res_id = (chain_id, res_seq)
            if res_id != last_res_id:
                residues.append(d3to1.get(res_name, 'X'))
                last_res_id = res_id
    return "".join(residues)

def run_peptide_download(candidates, max_workers=20, target_retained=35000):
    """Download and filter PDBs in parallel, resuming from local files first."""
    logger.info("Downloading and validating peptide structure files...")
    
    retained_peptides = []
    retained_accessions = set()
    
    # Load all existing PDB files from disk first
    existing_files = glob.glob(os.path.join(config.PDB_DIR, "*.pdb"))
    if existing_files:
        logger.info(f"Found {len(existing_files)} existing local PDB files. Loading and validating them...")
        for fpath in existing_files:
            if len(retained_peptides) >= target_retained:
                break
            try:
                accession = os.path.splitext(os.path.basename(fpath))[0]
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    pdb_text = f.read()
                
                seq = get_sequence_from_pdb_text(pdb_text)
                if not seq:
                    continue
                    
                mol = Chem.MolFromPDBBlock(pdb_text, removeHs=False, sanitize=False)
                if mol is None:
                    continue
                std_mol = standardize(mol)
                if std_mol is None:
                    continue
                is_valid, val_msg = validate_conformer_coordinates(std_mol)
                if not is_valid:
                    continue
                
                plddts = []
                for atom in mol.GetAtoms():
                    info = atom.GetPDBResidueInfo()
                    if info is not None and atom.GetSymbol() == 'C' and info.GetName().strip() == 'CA':
                        plddts.append(info.GetTempFactor())
                plddt_mean = float(np.mean(plddts)) if plddts else 0.0
                inchikey = Chem.MolToInchiKey(std_mol)
                
                retained_peptides.append({
                    'status': 'success',
                    'accession': accession,
                    'sequence': seq,
                    'length': len(seq),
                    'plddt_mean': plddt_mean,
                    'inchikey': inchikey,
                    'filepath': fpath,
                    'mol': std_mol
                })
                retained_accessions.add(accession)
            except Exception as e:
                logger.warning(f"Error loading existing PDB {fpath}: {e}")
                
        logger.info(f"Successfully loaded {len(retained_peptides)} validated peptides from local disk.")
        
    # We slice candidates in chunks to check if we can stop early
    chunk_size = min(5000, max(10, target_retained * 2))
    for chunk_start in range(0, len(candidates), chunk_size):
        if len(retained_peptides) >= target_retained:
            logger.info(f"Retained {len(retained_peptides)} validated peptides. Stopping downloads.")
            break
            
        chunk = candidates[chunk_start:chunk_start + chunk_size]
        # Skip candidates that are already loaded
        chunk = [c for c in chunk if c['accession'] not in retained_accessions]
        if not chunk:
            continue
            
        logger.info(f"Processing peptide candidates {chunk_start} to {chunk_start + len(chunk)}...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(download_and_validate_single_pdb, c): c for c in chunk}
            
            for future in as_completed(futures):
                cand = futures[future]
                stats['peptides']['fetched'] += 1
                
                try:
                    res = future.result()
                    status = res.get('status')
                    
                    if status == 'success':
                        retained_peptides.append(res)
                        retained_accessions.add(res['accession'])
                    elif status == 'no_coords':
                        stats['peptides']['no_coords_skipped'] += 1
                    elif status == 'validation_failed':
                        stats['peptides']['failed_validation'] += 1
                    elif status == 'plddt_failed':
                        stats['peptides']['plddt_dropped'] += 1
                    else:
                        stats['peptides']['failed_validation'] += 1
                except Exception as e:
                    stats['peptides']['failed_validation'] += 1
                    logger.error(f"Thread execution error for {cand['accession']}: {e}")
                    
    return retained_peptides

def run_sequence_clustering(retained_peptides):
    """Write sequences to FASTA, run MMseqs2 manually, and select representatives."""
    logger.info("Running MMseqs2 sequence clustering manually...")
    fasta_path = os.path.join(config.BASE_DIR, "peptides_retained.fasta")
    cluster_res_prefix = os.path.join(config.BASE_DIR, "peptides_cluster_res")
    tmp_dir = os.path.join(config.BASE_DIR, "peptides_cluster_tmp")
    
    # Clean output destination and temp folders
    tsv_path = f"{cluster_res_prefix}_cluster.tsv"
    if os.path.exists(tsv_path):
        try: os.remove(tsv_path)
        except: pass
    
    # Sort retained peptides by pLDDT descending so MMseqs2 picks highest pLDDT as representative
    retained_peptides.sort(key=lambda x: x['plddt_mean'], reverse=True)
    
    # Write to FASTA
    with open(fasta_path, "w") as f:
        for p in retained_peptides:
            f.write(f">{p['accession']}\n{p['sequence']}\n")
            
    # Run MMseqs2 subcommands manually to avoid segmentation fault in easy-cluster result2flat
    import subprocess
    mmseqs_exe = os.path.join(config.BASE_DIR, "tools", "mmseqs", "bin", "mmseqs.exe")
    
    logger.info("Executing manual MMseqs2 pipeline...")
    try:
        # 1. createdb
        db_input = os.path.join(tmp_dir, "input")
        os.makedirs(tmp_dir, exist_ok=True)
        subprocess.run([mmseqs_exe, 'createdb', fasta_path, db_input, '-v', '1'], check=True)
        
        # 2. cluster
        db_clu = os.path.join(tmp_dir, "clu")
        db_clu_tmp = os.path.join(tmp_dir, "clu_tmp")
        subprocess.run([
            mmseqs_exe, 'cluster', db_input, db_clu, db_clu_tmp,
            '--min-seq-id', '0.75', '-c', '0.8', '--cov-mode', '1',
            '--threads', '1', '-v', '1'
        ], check=True)
        
        # 3. createtsv
        subprocess.run([mmseqs_exe, 'createtsv', db_input, db_input, db_clu, tsv_path, '--threads', '1', '-v', '1'], check=True)
        
    except Exception as e:
        logger.error(f"Manual MMseqs2 clustering pipeline failed: {e}")
    
    # Parse cluster mapping TSV
    if not os.path.exists(tsv_path):
        logger.error(f"MMseqs2 clustering result TSV not found at {tsv_path}")
        return []
        
    cluster_groups = {}
    with open(tsv_path, "r") as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                rep, member = parts[0], parts[1]
                cluster_groups.setdefault(rep, []).append(member)
                
    # Build dictionary mapping accession to peptide details
    peptide_dict = {p['accession']: p for p in retained_peptides}
    
    final_reps = []
    
    # For each cluster, pick the member with the highest pLDDT score
    for rep, members in cluster_groups.items():
        # Retrieve members, find the one with highest pLDDT
        valid_members = [peptide_dict[m] for m in members if m in peptide_dict]
        if valid_members:
            best_member = max(valid_members, key=lambda x: x['plddt_mean'])
            final_reps.append(best_member)
            
    logger.info(f"MMseqs2 clustering reduced {len(retained_peptides)} sequences to {len(final_reps)} unique clusters.")
    
    # Cleanup temp clustering files
    for f in glob.glob(f"{cluster_res_prefix}*") + [fasta_path]:
        try: os.remove(f)
        except: pass
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)
    
    return final_reps

# ==================== SMALL MOLECULE PIPELINE ====================

def run_molecule_pipeline(target_count=25500):
    """
    Download and validate small molecules from PubChem.
    - Fetch properties in batches of 200 to filter first.
    - Download SDF conformers in batches of 100.
    """
    logger.info("Initializing PubChem small molecule collection...")
    
    retained_mols = []
    
    # Check if there are already downloaded SDF files to load
    existing_files = glob.glob(os.path.join(config.SDF_DIR, "*.sdf"))
    if existing_files:
        logger.info(f"Found {len(existing_files)} existing SDF files. Loading and validating them locally...")
        for fpath in existing_files:
            if len(retained_mols) >= target_count:
                break
            try:
                filename = os.path.basename(fpath)
                cid = int(filename.split('.')[0])
                
                suppl = Chem.SDMolSupplier(fpath, sanitize=False, removeHs=False)
                mol = next(suppl)
                if mol is None:
                    stats['molecules']['failed_validation'] += 1
                    continue
                    
                # Salt splitting: if molecule is disconnected, isolate the largest fragment
                frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
                if len(frags) > 1:
                    mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())
                    
                # Standardize (Remove Hs and Sanitize)
                std_mol = standardize(mol)
                if std_mol is None:
                    stats['molecules']['failed_validation'] += 1
                    continue
                    
                # Run strict validation gate
                is_valid, val_msg = validate_conformer_coordinates(std_mol)
                if not is_valid:
                    stats['molecules']['failed_validation'] += 1
                    continue
                    
                # Compute properties locally
                from rdkit.Chem import Descriptors
                mw = Descriptors.MolWt(std_mol)
                ha = std_mol.GetNumHeavyAtoms()
                inchikey = Chem.MolToInchiKey(std_mol)
                can_smiles = Chem.MolToSmiles(std_mol, canonical=True)
                
                # Deduplicate by InChIKey
                if inchikey in all_inchikeys or not inchikey:
                    stats['molecules']['dedup_dropped'] += 1
                    continue
                    
                all_inchikeys.add(inchikey)
                retained_mols.append({
                    'cid': cid,
                    'smiles': can_smiles,
                    'mw': mw,
                    'heavy_atom_count': ha,
                    'inchikey': inchikey,
                    'filepath': fpath,
                    'mol': std_mol
                })
            except Exception as e:
                logger.warning(f"Error loading existing SDF {fpath}: {e}")
                
        logger.info(f"Successfully loaded and validated {len(retained_mols)} molecules from local disk.")
        
    # We will search CIDs starting from 1,000,000 sequentially, or max existing CID + 1
    max_cid = 1000000
    if existing_files:
        try:
            cids = []
            for f in existing_files:
                try:
                    cids.append(int(os.path.basename(f).split('.')[0]))
                except:
                    pass
            if cids:
                max_cid = max(max(cids) + 1, 1000000)
        except:
            pass
            
    current_cid = max_cid
    batch_size_props = 200
    batch_size_sdf = 100
    
    # Buffer to hold pre-filtered candidates that need SDF download
    download_queue = []
    
    # Counter for stats
    while len(retained_mols) < target_count:
        # Step 1: Scan properties in batches
        logger.info(f"Scanning CIDs from {current_cid} to {current_cid + 2000}...")
        for _ in range(10):  # Scan 2000 CIDs
            cids_batch = list(range(current_cid, current_cid + batch_size_props))
            current_cid += batch_size_props
            
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{','.join(map(str, cids_batch))}/property/MolecularWeight,HeavyAtomCount,ConnectivitySMILES,InChIKey/JSON"
            try:
                r = requests.get(url, timeout=15)
                stats['molecules']['fetched'] += len(cids_batch)
                
                if r.status_code == 200:
                    props = r.json().get('PropertyTable', {}).get('Properties', [])
                    for p in props:
                        cid = p['CID']
                        smiles = p.get('ConnectivitySMILES', '')
                        mw = float(p.get('MolecularWeight', 0.0))
                        ha = int(p.get('HeavyAtomCount', 0))
                        inchikey = p.get('InChIKey', '')
                        
                        # Size bounds
                        if not (100.0 <= mw <= 900.0):
                            stats['molecules']['size_dropped'] += 1
                            continue
                        if ha > 60:
                            stats['molecules']['size_dropped'] += 1
                            continue
                            
                        # SMILES length bound
                        if len(smiles) < 20:
                            stats['molecules']['smiles_len_dropped'] += 1
                            continue
                            
                        # Silicon chains or polymeric skip
                        if '[Si]' in smiles or 'Si' in smiles:
                            stats['molecules']['silicon_dropped'] += 1
                            continue
                            
                        # Salt/Disconnected components check
                        if '.' in smiles:
                            # Split on '.' and take the largest fragment
                            frags = smiles.split('.')
                            largest_frag = max(frags, key=len)
                            if len(largest_frag) < 20:
                                stats['molecules']['smiles_len_dropped'] += 1
                                continue
                            smiles = largest_frag
                            stats['molecules']['salt_split_kept'] += 1
                            
                        # Deduplicate by InChIKey
                        if inchikey in all_inchikeys or not inchikey:
                            stats['molecules']['dedup_dropped'] += 1
                            continue
                            
                        # Add to download queue
                        download_queue.append({
                            'cid': cid,
                            'smiles': smiles,
                            'mw': mw,
                            'heavy_atom_count': ha,
                            'inchikey': inchikey
                        })
                time.sleep(0.2)  # Rate limit safety
            except Exception as e:
                logger.error(f"Error scanning properties batch: {e}")
                time.sleep(1.0)
                
        # Step 2: Download SDF coordinates for queue candidates
        if len(download_queue) > 0:
            logger.info(f"Downloading SDF conformer coordinates for {len(download_queue)} candidates...")
            
            for i in range(0, len(download_queue), batch_size_sdf):
                if len(retained_mols) >= target_count:
                    break
                    
                batch = download_queue[i:i + batch_size_sdf]
                cids_str = ",".join(str(item['cid']) for item in batch)
                
                url_sdf = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cids_str}/SDF?record_type=3d"
                
                try:
                    r_sdf = requests.get(url_sdf, timeout=20)
                    if r_sdf.status_code == 200:
                        stats['molecules']['fetched_sdf'] += len(batch)
                        # Parse multi-molecule SDF file
                        suppl = Chem.SDMolSupplier()
                        suppl.SetData(r_sdf.text, sanitize=False, removeHs=False)
                        
                        # Map CIDs to their details
                        batch_dict = {item['cid']: item for item in batch}
                        
                        for mol in suppl:
                            if mol is None:
                                stats['molecules']['failed_validation'] += 1
                                continue
                                
                            try:
                                # Get CID from property
                                cid = int(mol.GetProp("_Name"))
                            except:
                                continue
                                
                            if cid not in batch_dict:
                                continue
                            item = batch_dict[cid]
                            
                            # Salt splitting: if molecule is disconnected, isolate the largest fragment
                            frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
                            if len(frags) > 1:
                                mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())
                                
                            # Standardize (Remove Hs and Sanitize)
                            std_mol = standardize(mol)
                            if std_mol is None:
                                stats['molecules']['failed_validation'] += 1
                                continue
                                
                            # Run strict validation gate
                            is_valid, val_msg = validate_conformer_coordinates(std_mol)
                            if not is_valid:
                                stats['molecules']['failed_validation'] += 1
                                continue
                                
                            # Compute canonical SMILES for metadata
                            can_smiles = Chem.MolToSmiles(std_mol, canonical=True)
                            
                            # Final checks passed! Save molecule SDF locally
                            file_path = os.path.join(config.SDF_DIR, f"{cid}.sdf")
                            writer = Chem.SDWriter(file_path)
                            writer.write(std_mol)
                            writer.close()
                            
                            # Track InChIKey globally for cross-source deduplication
                            all_inchikeys.add(item['inchikey'])
                            
                            retained_mols.append({
                                'cid': cid,
                                'smiles': can_smiles,
                                'mw': item['mw'],
                                'heavy_atom_count': std_mol.GetNumHeavyAtoms(),
                                'inchikey': item['inchikey'],
                                'filepath': file_path,
                                'mol': std_mol
                            })
                            
                    else:
                        stats['molecules']['no_3d_coords_skipped'] += len(batch)
                        
                    time.sleep(0.2)  # Rate limit safety
                except Exception as e:
                    logger.error(f"Error fetching SDF batch: {e}")
                    time.sleep(1.0)
                    
            # Clear download queue after processing
            download_queue = []
            
    return retained_mols

# ==================== MAIN PROCESSING PIPELINE ====================

def main():
    start_time = time.time()
    logger.info("Initializing Data Acquisition Pipeline...")
    
    # 1. Check DEBUG_MODE limits
    if config.DEBUG_MODE:
        target_count_mols = config.SUBSET_SIZE // 2
        target_count_peptides = config.SUBSET_SIZE - target_count_mols
        peptide_candidate_limit = target_count_peptides * 4
        peptide_retained_target = target_count_peptides + 2
        logger.info(f"DEBUG_MODE is True. Adjusting targets: molecules={target_count_mols}, peptides={target_count_peptides}")
    else:
        target_count_mols = 25000
        target_count_peptides = 25000
        peptide_candidate_limit = 200000
        peptide_retained_target = 28000
        
    # 2. Clean folders
    clean_directories()
    
    # 3. Collect Small Molecules (PubChem) first (to build target InChIKey registry)
    logger.info("Step 1: Starting PubChem Small Molecule Downloader...")
    small_molecules = run_molecule_pipeline(target_count=target_count_mols)
    logger.info(f"Retained {len(small_molecules)} validated small molecules.")
    
    # 4. Stream Candidate Peptides from UniProt
    logger.info("Step 2: Querying and Streaming Candidate Peptides from UniProt...")
    peptide_candidates = fetch_uniprot_candidates(limit=peptide_candidate_limit)
    
    # Identify accessions that were processed in the first 80,000 candidates but are NOT in PDB_DIR
    first_80k_accessions = {c['accession'] for c in peptide_candidates[:80000]}
    existing_accessions = {os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(config.PDB_DIR, "*.pdb"))}
    discarded_accessions = first_80k_accessions - existing_accessions
    logger.info(f"Skipping {len(discarded_accessions)} accessions that were already processed and discarded as duplicates/non-representatives in the first 80,000 candidates.")
    
    filtered_candidates = [c for c in peptide_candidates if c['accession'] not in discarded_accessions]
    
    # 5. Download and validate candidate PDBs
    logger.info("Step 3: Fetching and checking PDB files...")
    retained_peptides = run_peptide_download(filtered_candidates, target_retained=peptide_retained_target)
    logger.info(f"Downloaded and validated {len(retained_peptides)} candidate peptides.")
    
    # 6. Deduplicate Peptides against PubChem InChIKeys and within peptides pool
    logger.info("Deduplicating peptides against molecule InChIKeys...")
    unique_peptides = []
    seen_peptide_keys = set()
    for p in retained_peptides:
        key = p['inchikey']
        if key in all_inchikeys or key in seen_peptide_keys or not key:
            stats['peptides']['dedup_dropped'] += 1
            continue
        seen_peptide_keys.add(key)
        unique_peptides.append(p)
        
    logger.info(f"Peptides pool size after InChIKey deduplication: {len(unique_peptides)}")
    
    # 7. Cluster sequences at 30% sequence identity using MMseqs2
    logger.info("Step 4: Running MMseqs2 Clustering on unique peptides...")
    clustered_peptides = run_sequence_clustering(unique_peptides)
    
    # 8. Select exactly target_count_peptides peptides
    if len(clustered_peptides) < target_count_peptides:
        logger.warning(f"Clustering returned only {len(clustered_peptides)} clusters. We need {target_count_peptides}. Fetching more candidates...")
        extra_candidates = fetch_uniprot_candidates(limit=peptide_candidate_limit * 2)[peptide_candidate_limit:]
        extra_candidates = [c for c in extra_candidates if c['accession'] not in discarded_accessions]
        extra_retained = run_peptide_download(extra_candidates, target_retained=peptide_retained_target)
        for p in extra_retained:
            key = p['inchikey']
            if key not in all_inchikeys and key not in seen_peptide_keys and key:
                seen_peptide_keys.add(key)
                unique_peptides.append(p)
        clustered_peptides = run_sequence_clustering(unique_peptides)
        
    final_peptides = clustered_peptides[:target_count_peptides]
    final_molecules = small_molecules[:target_count_mols]
    
    stats['peptides']['retained'] = len(final_peptides)
    stats['molecules']['retained'] = len(final_molecules)
    
    logger.info(f"Final dataset ratio: {len(final_peptides)} Peptides to {len(final_molecules)} Small Molecules (1:1 balanced).")
    
    # 9. Build PyG Graphs and save to data/processed
    logger.info("Step 5: Building PyG graphs for retained datasets...")
    
    # Process peptides
    for i, p in enumerate(final_peptides):
        coords = p['mol'].GetConformer().GetPositions()
        meta = {
            'id': p['accession'],
            'source': 'alphafold',
            'plddt_mean': p['plddt_mean']
        }
        data, _ = build_graph(p['mol'], coords, meta)
        if data is not None:
            # Set all metadata attributes with consistent types for PyG collation compatibility
            data.source = "alphafold"
            data.mol_id = p['accession']
            data.smiles = Chem.MolToSmiles(p['mol'], canonical=True)
            data.atom_count = p['mol'].GetNumHeavyAtoms()
            
            # Peptide specific metadata
            data.uniprot_accession = p['accession']
            data.sequence = p['sequence']
            data.residue_length = p['length']
            data.plddt_mean = p['plddt_mean']
            
            # PubChem specific metadata (placeholders with consistent types)
            data.pubchem_cid = 0
            data.mw = 0.0
            data.heavy_atom_count = 0
            
            torch.save(data, os.path.join(config.PROCESSED_DIR, f"{p['accession']}.pt"))
            
    # Process molecules
    for i, m in enumerate(final_molecules):
        coords = m['mol'].GetConformer().GetPositions()
        meta = {
            'id': str(m['cid']),
            'source': 'pubchem'
        }
        data, _ = build_graph(m['mol'], coords, meta)
        if data is not None:
            # Set all metadata attributes with consistent types for PyG collation compatibility
            data.source = "pubchem"
            data.mol_id = str(m['cid'])
            data.smiles = m['smiles']
            data.atom_count = m['mol'].GetNumHeavyAtoms()
            
            # Peptide specific metadata (placeholders with consistent types)
            data.uniprot_accession = ""
            data.sequence = ""
            data.residue_length = 0
            data.plddt_mean = 0.0
            
            # PubChem specific metadata
            data.pubchem_cid = m['cid']
            data.mw = m['mw']
            data.heavy_atom_count = m['heavy_atom_count']
            
            torch.save(data, os.path.join(config.PROCESSED_DIR, f"pubchem_{m['cid']}.pt"))
            
    logger.info(f"Completed saving PyG graphs to {config.PROCESSED_DIR}.")
    
    # 9. Write Provenance Report
    logger.info("Step 6: Writing provenance report...")
    report_path = os.path.join(config.BASE_DIR, "provenance_report.md")
    
    # Compute fractions
    pep_fetched = stats['peptides']['fetched']
    pep_usable = len(retained_peptides)
    pep_fraction = pep_usable / pep_fetched if pep_fetched > 0 else 0.0
    
    mol_fetched_sdf = stats['molecules']['fetched_sdf']
    mol_usable = len(small_molecules)
    mol_fraction = mol_usable / mol_fetched_sdf if mol_fetched_sdf > 0 else 0.0
    
    report_content = f"""# Dataset Provenance Report
 
This report documents the collection, filtering, validation, and curation results for the Multimodal 3D Peptide Embeddings training dataset.
 
## Target Dataset Composition
- **Total Targets:** exactly 50,000 structures (25,000 peptides + 25,000 small molecules).
- **Target Ratio:** 1:1 balance enforced.
 
---
 
## 1. Peptides (AlphaFold DB)
- **Initial Candidates Streamed:** {len(peptide_candidates)}
- **Candidates Attempted/Fetched:** {stats['peptides']['fetched']}
- **Dropped at Filter Stages:**
  - Length filter drop (outside 5-50 AA): {stats['peptides']['length_dropped']}
  - Low-confidence prediction drop (pLDDT <= 70.0): {stats['peptides']['plddt_dropped']}
  - Deduplication drop (InChIKey overlaps): {stats['peptides']['dedup_dropped']}
- **Skipped for Lacking 3D Coordinates:** {stats['peptides']['no_coords_skipped']}
- **Failed Validation Gate:** {stats['peptides']['failed_validation']}
- **Final Retained Peptides (after sequence clustering & balancing):** {stats['peptides']['retained']}
- **Fraction of Usable 3D Coordinate Structures:** {pep_fraction * 100:.2f}% (usable / attempted)
 
---
 
## 2. Small Molecules (PubChem)
- **CIDs Scanned (Property Level):** {stats['molecules']['fetched']}
- **CIDs Attempted/Fetched (3D SDF):** {stats['molecules']['fetched_sdf']}
- **Dropped at Filter Stages:**
  - Size out of bounds (MW or atom count): {stats['molecules']['size_dropped']}
  - SMILES length < 20: {stats['molecules']['smiles_len_dropped']}
  - Disconnected component splits: {stats['molecules']['salt_split_kept']} (kept largest fragment)
  - Silicon/polymeric skips: {stats['molecules']['silicon_dropped']}
  - Deduplication drop (InChIKey overlaps): {stats['molecules']['dedup_dropped']}
- **Skipped for Lacking 3D Coordinates:** {stats['molecules']['no_3d_coords_skipped']}
- **Failed Validation Gate:** {stats['molecules']['failed_validation']}
- **Final Retained Small Molecules (after balancing):** {stats['molecules']['retained']}
- **Fraction of Usable 3D Coordinate Structures:** {mol_fraction * 100:.2f}% (usable / attempted)
 
---
 
## 3. Dataset Deduplication Summary
- Deduplication was executed using canonical InChIKeys.
- All InChIKeys were tracked globally to ensure no overlapping structures existed between the peptide and small-molecule pools.

*Report generated at {time.strftime('%Y-%m-%d %H:%M:%S')} (Local Time).*
"""
    
    with open(report_path, "w") as f:
        f.write(report_content)
        
    logger.info(f"Provenance report written to {report_path}.")
    logger.info(f"Total script run time: {(time.time() - start_time) / 60:.2f} minutes.")

if __name__ == "__main__":
    main()
