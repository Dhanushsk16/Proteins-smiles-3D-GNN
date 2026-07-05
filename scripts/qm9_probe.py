import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import QM9
from torch_geometric.data import Data
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from model import Encoder
from dataset import RadiusGraphTransform

ATOMIC_NUM_TO_SYMBOL = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 16: "S", 15: "P", 17: "Cl", 35: "Br", 53: "I"}


def atomic_num_to_index(z_val):
    symbol = ATOMIC_NUM_TO_SYMBOL.get(z_val, None)
    if symbol is not None and symbol in config.ALLOWED_ELEMENTS:
        return config.ALLOWED_ELEMENTS.index(symbol)
    return len(config.ALLOWED_ELEMENTS)


def load_frozen_encoder(checkpoint_path, device):
    encoder = Encoder()
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    encoder.load_state_dict(state_dict)
    encoder.to(device)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad_(False)
    print(f"Loaded encoder from {checkpoint_path}")
    num_params = sum(p.numel() for p in encoder.parameters())
    print(f"Encoder parameters: {num_params:,} (all frozen)")
    return encoder


def convert_qm9_to_pipeline_format(qm9_data):
    x = torch.tensor([atomic_num_to_index(int(z)) for z in qm9_data.z], dtype=torch.long)
    converted = Data(x=x, pos=qm9_data.pos)
    return converted


def sanity_check_encoder(encoder, qm9_dataset, device):
    print("\n--- Step 0: Encoder Sanity Check ---")
    rg = RadiusGraphTransform()
    test_graphs = []
    for i in range(min(5, len(qm9_dataset))):
        g = convert_qm9_to_pipeline_format(qm9_dataset[i])
        g = rg(g)
        test_graphs.append(g)

    loader = DataLoader(test_graphs, batch_size=len(test_graphs))
    batch = next(iter(loader)).to(device)
    with torch.no_grad():
        atom_vecs, pooled = encoder(batch.x, batch.pos, batch.edge_index, batch.edge_attr, batch.batch)

    expected_dim = config.HIDDEN_DIM
    assert pooled.shape == (len(test_graphs), expected_dim), \
        f"Expected pooled shape [{len(test_graphs)}, {expected_dim}], got {pooled.shape}"
    assert torch.isfinite(pooled).all(), "Pooled embeddings contain NaN or Inf — checkpoint is bad"
    assert pooled.std() > 1e-6, "Pooled embeddings are near-constant — checkpoint is bad"

    print(f"  Pooled shape: {pooled.shape} (expected [{len(test_graphs)}, {expected_dim}])")
    print(f"  Embedding mean: {pooled.mean().item():.4f}, std: {pooled.std().item():.4f}")
    print(f"  All finite: True, Non-degenerate: True")
    print("  Sanity check PASSED")


def element_coverage_check(qm9_dataset):
    print("\n--- Step 1: Element Coverage Check ---")
    other_idx = len(config.ALLOWED_ELEMENTS)
    other_count = 0
    element_counts = defaultdict(int)

    for i in range(len(qm9_dataset)):
        for z_val in qm9_dataset[i].z.tolist():
            idx = atomic_num_to_index(int(z_val))
            if idx == other_idx:
                other_count += 1
            symbol = ATOMIC_NUM_TO_SYMBOL.get(int(z_val), f"Z={z_val}")
            element_counts[symbol] += 1

    print(f"  Elements found: {dict(element_counts)}")
    print(f"  Atoms hitting 'other' catch-all: {other_count}")
    assert other_count == 0, \
        f"FATAL: {other_count} atoms mapped to 'other' — atom index mapping is broken, probe results would be invalid"
    print("  Coverage check PASSED (all QM9 atoms map to explicit vocabulary slots)")


def extract_embeddings(encoder, qm9_dataset, device, batch_size=256):
    print("\n--- Step 3: Extracting Frozen Embeddings ---")
    rg = RadiusGraphTransform()
    all_graphs = []
    targets = []

    for i in range(len(qm9_dataset)):
        raw = qm9_dataset[i]
        g = convert_qm9_to_pipeline_format(raw)
        g = rg(g)
        all_graphs.append(g)
        targets.append(raw.y[0, config.QM9_TARGET_IDX].item())

    targets = np.array(targets)
    loader = DataLoader(all_graphs, batch_size=batch_size, shuffle=False)

    all_embeddings = []
    with torch.no_grad():
        for batch_data in loader:
            batch_data = batch_data.to(device)
            _, pooled = encoder(batch_data.x, batch_data.pos, batch_data.edge_index, batch_data.edge_attr, batch_data.batch)
            all_embeddings.append(pooled.cpu().numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)
    print(f"  Extracted embeddings: {embeddings.shape}")
    print(f"  Target ({config.QM9_TARGET_NAME}): mean={targets.mean():.4f}, std={targets.std():.4f} {config.QM9_TARGET_UNIT}")
    return embeddings, targets


def get_smiles_list(qm9_dataset):
    smiles_list = []
    for i in range(len(qm9_dataset)):
        smi = qm9_dataset[i].smiles if hasattr(qm9_dataset[i], 'smiles') else None
        if smi is None:
            smi = ""
        smiles_list.append(smi)
    return smiles_list


def scaffold_split(smiles_list, target_fracs=(0.8, 0.1, 0.1), seed=42):
    print("\n--- Step 4: Scaffold Split (DeepChem-style, Unbiased) ---")
    scaffold_to_indices = defaultdict(list)

    for i, smi in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
            else:
                scaffold = f"__failed_{i}"
        except Exception:
            scaffold = f"__failed_{i}"
        scaffold_to_indices[scaffold].append(i)

    scaffold_groups = list(scaffold_to_indices.values())
    scaffold_groups.sort(key=len, reverse=True)

    n = len(smiles_list)
    target_sizes = [int(f * n) for f in target_fracs]
    target_sizes[0] = n - target_sizes[1] - target_sizes[2]

    train_idx, val_idx, test_idx = [], [], []
    train_cutoff, val_cutoff, test_cutoff = target_sizes

    rng = np.random.RandomState(seed)

    for group in scaffold_groups:
        train_deficit = train_cutoff - len(train_idx)
        val_deficit = val_cutoff - len(val_idx)
        test_deficit = test_cutoff - len(test_idx)

        # Assign to a split randomly, weighted by the remaining deficit
        deficits = [max(0, train_deficit), max(0, val_deficit), max(0, test_deficit)]
        sum_deficits = sum(deficits)
        if sum_deficits == 0:
            probs = list(target_fracs)
        else:
            probs = [d / sum_deficits for d in deficits]

        split = rng.choice(["train", "val", "test"], p=probs)
        if split == "train":
            train_idx.extend(group)
        elif split == "val":
            val_idx.extend(group)
        else:
            test_idx.extend(group)

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    print(f"  Total molecules: {n}")
    print(f"  Unique scaffolds: {len(scaffold_groups)}")
    print(f"  Train: {len(train_idx)} ({100*len(train_idx)/n:.1f}%)")
    print(f"  Val:   {len(val_idx)} ({100*len(val_idx)/n:.1f}%)")
    print(f"  Test:  {len(test_idx)} ({100*len(test_idx)/n:.1f}%)")
    return train_idx, val_idx, test_idx


def compute_morgan_fingerprints(smiles_list, radius=2, n_bits=2048):
    print("\n  Computing Morgan fingerprints...")
    fps = np.zeros((len(smiles_list), n_bits), dtype=np.float32)
    for i, smi in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
                fps[i] = np.array(fp)
        except Exception:
            pass
    return fps


class MLPProbe(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp_probe(X_train, y_train, X_val, y_val, input_dim, seed, max_epochs=100, patience=10):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MLPProbe(input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.float32)
    X_v = torch.tensor(X_val, dtype=torch.float32)
    y_v = torch.tensor(y_val, dtype=torch.float32)

    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        preds = model(X_tr)
        loss = nn.functional.mse_loss(preds, y_tr)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_preds = model(X_v)
            val_loss = nn.functional.mse_loss(val_preds, y_v).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_state)
    return model


def evaluate_metrics(y_true, y_pred, target_mean, target_std):
    y_true_orig = y_true * target_std + target_mean
    y_pred_orig = y_pred * target_std + target_mean
    mae = mean_absolute_error(y_true_orig, y_pred_orig)
    rmse = np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))
    return mae, rmse


def run_probes(embeddings, targets, fingerprints, train_idx, val_idx, test_idx):
    print("\n--- Step 5: Fitting Probes ---")

    target_mean = targets[train_idx].mean()
    target_std = targets[train_idx].std()
    targets_std = (targets - target_mean) / target_std

    X_train_emb = embeddings[train_idx]
    X_val_emb = embeddings[val_idx]
    X_test_emb = embeddings[test_idx]
    y_train = targets_std[train_idx]
    y_val = targets_std[val_idx]
    y_test = targets_std[test_idx]

    X_train_fp = fingerprints[train_idx]
    X_test_fp = fingerprints[test_idx]

    results = {}

    print("\n  [1/4] Mean predictor (floor)...")
    mean_pred = np.zeros_like(y_test)
    mae, rmse = evaluate_metrics(y_test, mean_pred, target_mean, target_std)
    results["Mean predictor (floor)"] = {"mae": mae, "rmse": rmse, "note": ""}
    print(f"    Test MAE: {mae:.4f} {config.QM9_TARGET_UNIT} | Test RMSE: {rmse:.4f} {config.QM9_TARGET_UNIT}")

    print("\n  [2/4] Frozen embeddings + Ridge (linear probe)...")
    ridge_emb = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
    ridge_emb.fit(X_train_emb, y_train)
    preds_emb = ridge_emb.predict(X_test_emb)
    mae, rmse = evaluate_metrics(y_test, preds_emb, target_mean, target_std)
    results["Frozen embeddings + Linear"] = {"mae": mae, "rmse": rmse, "note": f"alpha={ridge_emb.alpha_:.2f}"}
    print(f"    Test MAE: {mae:.4f} {config.QM9_TARGET_UNIT} | Test RMSE: {rmse:.4f} {config.QM9_TARGET_UNIT} | alpha={ridge_emb.alpha_:.2f}")

    print("\n  [3/4] Frozen embeddings + MLP (3 seeds)...")
    mlp_maes, mlp_rmses = [], []
    for seed in [42, 123, 456]:
        model = train_mlp_probe(X_train_emb, y_train, X_val_emb, y_val, config.HIDDEN_DIM, seed)
        model.eval()
        with torch.no_grad():
            preds_mlp = model(torch.tensor(X_test_emb, dtype=torch.float32)).numpy()
        mae_s, rmse_s = evaluate_metrics(y_test, preds_mlp, target_mean, target_std)
        mlp_maes.append(mae_s)
        mlp_rmses.append(rmse_s)
    mae_mean, mae_std = np.mean(mlp_maes), np.std(mlp_maes)
    rmse_mean, rmse_std = np.mean(mlp_rmses), np.std(mlp_rmses)
    results["Frozen embeddings + MLP"] = {
        "mae": mae_mean, "rmse": rmse_mean,
        "note": f"+/-{mae_std:.4f} MAE, +/-{rmse_std:.4f} RMSE (3 seeds)"
    }
    print(f"    Test MAE: {mae_mean:.4f}+/-{mae_std:.4f} {config.QM9_TARGET_UNIT} | Test RMSE: {rmse_mean:.4f}+/-{rmse_std:.4f} {config.QM9_TARGET_UNIT}")

    print("\n  [4/4] Morgan fingerprint + Ridge (baseline)...")
    ridge_fp = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
    y_train_fp = targets_std[train_idx]
    y_test_fp = targets_std[test_idx]
    ridge_fp.fit(X_train_fp, y_train_fp)
    preds_fp = ridge_fp.predict(X_test_fp)
    mae, rmse = evaluate_metrics(y_test_fp, preds_fp, target_mean, target_std)
    results["Morgan fingerprint + Linear (baseline)"] = {"mae": mae, "rmse": rmse, "note": f"alpha={ridge_fp.alpha_:.2f}"}
    print(f"    Test MAE: {mae:.4f} {config.QM9_TARGET_UNIT} | Test RMSE: {rmse:.4f} {config.QM9_TARGET_UNIT} | alpha={ridge_fp.alpha_:.2f}")

    return results


def print_results_table(results):
    print("\n\n" + "=" * 80)
    print(f"RESULTS: Frozen-Embedding Probe on QM9 ({config.QM9_TARGET_NAME})")
    print("=" * 80)
    print(f"\nTarget: {config.QM9_TARGET_NAME} (index {config.QM9_TARGET_IDX})")
    print(f"Units: {config.QM9_TARGET_UNIT}")
    print(f"Split: Scaffold (Bemis-Murcko, DeepChem-style)")
    print(f"Encoder: SchNet, checkpoint = {os.path.basename(config.ENCODER_CHECKPOINT)}")
    print(f"Embedding dim: {config.HIDDEN_DIM}")
    print()

    header = f"| {'Probe':<42} | {'Test MAE':>10} | {'Test RMSE':>10} | {'Notes':<35} |"
    sep = f"|{'-'*43}|{'-'*12}|{'-'*12}|{'-'*37}|"
    print(header)
    print(sep)
    for name, m in results.items():
        unit = config.QM9_TARGET_UNIT
        mae_str = f"{m['mae']:.4f} {unit}"
        rmse_str = f"{m['rmse']:.4f} {unit}"
        print(f"| {name:<42} | {mae_str:>10} | {rmse_str:>10} | {m['note']:<35} |")
    print()

    emb_linear = results.get("Frozen embeddings + Linear", {})
    fp_baseline = results.get("Morgan fingerprint + Linear (baseline)", {})
    if emb_linear and fp_baseline:
        if emb_linear["mae"] < fp_baseline["mae"]:
            print("CONCLUSION: Frozen embeddings BEAT the fingerprint baseline.")
            print("  -> Pretraining added structural signal beyond trivial counting.")
        else:
            print("CONCLUSION: Frozen embeddings did NOT beat the fingerprint baseline.")
            print("  -> Pretraining may not have captured sufficient task-relevant signal,")
            print("    or the linear probe cannot extract it.")


def main():
    device = torch.device(config.DEVICE)
    print(f"Running on: {device}")
    print(f"Encoder checkpoint: {config.ENCODER_CHECKPOINT}")

    assert os.path.exists(config.ENCODER_CHECKPOINT), \
        f"Checkpoint not found: {config.ENCODER_CHECKPOINT}"

    encoder = load_frozen_encoder(config.ENCODER_CHECKPOINT, device)

    print("\n--- Step 1: Loading QM9 ---")
    qm9_dir = config.QM9_DIR
    raw_dir = os.path.join(qm9_dir, "raw")
    processed_dir = os.path.join(qm9_dir, "processed")
    os.makedirs(raw_dir, exist_ok=True)

    # Automatically download uncharacterized.txt if it's missing from raw_dir
    unchar_path = os.path.join(raw_dir, "uncharacterized.txt")
    if not os.path.exists(unchar_path):
        print("  Downloading uncharacterized.txt from GitHub mirror...")
        import requests
        url = "https://raw.githubusercontent.com/bondrewd/dataset-qm9-raw/master/uncharacterized.txt"
        try:
            r = requests.get(url)
            r.raise_for_status()
            with open(unchar_path, "wb") as f:
                f.write(r.content)
            print("  Downloaded uncharacterized.txt successfully.")
        except Exception as e:
            print(f"  Failed to download uncharacterized.txt from GitHub: {e}")
            sys.exit(1)

    # Check if processed dataset lacks SMILES (from previous failed rdkit runs) and clean if so
    processed_file = os.path.join(processed_dir, "data_v3.pt")
    if os.path.exists(processed_file):
        try:
            qm9_test = QM9(root=qm9_dir)
            if len(qm9_test) > 0 and not hasattr(qm9_test[0], "smiles"):
                print("  Detected processed dataset without SMILES. Deleting processed folder to trigger correct RDKit processing...")
                for f in os.listdir(processed_dir):
                    if f.endswith(".pt"):
                        os.remove(os.path.join(processed_dir, f))
        except Exception as e:
            print(f"  Error checking processed dataset: {e}. Clearing processed folder to be safe...")
            if os.path.exists(processed_dir):
                for f in os.listdir(processed_dir):
                    if f.endswith(".pt"):
                        os.remove(os.path.join(processed_dir, f))

    qm9 = QM9(root=qm9_dir)
    print(f"  QM9 loaded: {len(qm9)} molecules")

    element_coverage_check(qm9)
    sanity_check_encoder(encoder, qm9, device)

    embeddings, targets = extract_embeddings(encoder, qm9, device)

    smiles_list = get_smiles_list(qm9)
    train_idx, val_idx, test_idx = scaffold_split(smiles_list)

    fingerprints = compute_morgan_fingerprints(smiles_list)

    results = run_probes(embeddings, targets, fingerprints, train_idx, val_idx, test_idx)
    print_results_table(results)


if __name__ == "__main__":
    main()
