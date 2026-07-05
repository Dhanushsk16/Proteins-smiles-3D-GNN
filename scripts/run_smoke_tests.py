import os
import sys
import time
import psutil
import numpy as np
import torch
from torch_geometric.loader import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from dataset import MoleculeGraphDataset, CoordinateNoiseTransform
import model

class CleanDataTransform:
    def __call__(self, data):
        keys_to_keep = {'x', 'pos', 'edge_index', 'edge_attr', 'dist_target'}
        for key in list(data.keys()):
            if key not in keys_to_keep:
                delattr(data, key)
        if data.x.ndim == 2:
            data.x = data.x[:, :11].argmax(dim=-1)
        return data

class JointTransform:
    def __init__(self):
        self.noise = CoordinateNoiseTransform()
        self.clean = CleanDataTransform()
    def __call__(self, data):
        data = self.noise(data)
        data = self.clean(data)
        return data

def compute_triplets(edge_index, num_nodes):
    row, col = edge_index
    match = (col.unsqueeze(1) == row.unsqueeze(0))
    not_back = (row.unsqueeze(1) != col.unsqueeze(0))
    valid = match & not_back
    idx_kj, idx_ji = torch.nonzero(valid, as_tuple=True)
    i = col[idx_ji]
    j = col[idx_kj]
    k = row[idx_kj]
    return i, j, i, j, k, idx_kj, idx_ji

def run_shape_test(model_name, sample_batch, device):
    config.ENCODER_TYPE = model_name
    net = model.PretrainModel().to(device)
    net.eval()
    data = sample_batch.to(device)
    with torch.no_grad():
        atom_vectors, mol_embedding = net.encoder(data.x, data.pos, data.edge_index, data.edge_attr, data.batch)
    
    assert atom_vectors.ndim == 2
    assert atom_vectors.shape[0] == data.x.shape[0]
    assert atom_vectors.shape[1] == config.HIDDEN_DIM
    assert mol_embedding.ndim == 2
    assert mol_embedding.shape[0] == sample_batch.num_graphs
    assert mol_embedding.shape[1] == config.HIDDEN_DIM
    assert not torch.isnan(atom_vectors).any()
    assert not torch.isnan(mol_embedding).any()
    return "Pass"

def run_equivariance_test(model_name, sample_batch, device):
    config.ENCODER_TYPE = model_name
    net = model.PretrainModel().to(device)
    net.eval()
    data = sample_batch.to(device)
    
    q, r = torch.linalg.qr(torch.randn(3, 3, device=device))
    R = q * torch.det(q).sign()
    t = torch.randn(1, 3, device=device)
    pos_rot = torch.matmul(data.pos, R.t()) + t
    
    with torch.no_grad():
        atom_vectors_clean, mol_embedding_clean = net.encoder(
            data.x, data.pos, data.edge_index, data.edge_attr, data.batch
        )
        atom_vectors_rot, mol_embedding_rot = net.encoder(
            data.x, pos_rot, data.edge_index, data.edge_attr, data.batch
        )
        
    inv_pass = torch.allclose(mol_embedding_clean, mol_embedding_rot, atol=1e-5, rtol=1e-5)
    node_inv_pass = torch.allclose(atom_vectors_clean, atom_vectors_rot, atol=1e-5, rtol=1e-5)
    
    equiv_pass = True
    if model_name == "egnn":
        feats = net.encoder.node_emb(data.x)
        inp_clean = torch.cat([data.pos, feats], dim=-1)
        inp_rot = torch.cat([pos_rot, feats], dim=-1)
        with torch.no_grad():
            out_clean = inp_clean
            out_rot = inp_rot
            for layer in net.encoder.layers:
                out_clean = layer(out_clean, data.edge_index, batch=data.batch)
                out_rot = layer(out_rot, data.edge_index, batch=data.batch)
            coors_clean = out_clean[:, :3]
            coors_rot = out_rot[:, :3]
            coors_clean_transformed = torch.matmul(coors_clean, R.t()) + t
            equiv_pass = torch.allclose(coors_rot, coors_clean_transformed, atol=1e-5, rtol=1e-5)
            
    elif model_name == "painn":
        s = net.encoder.node_emb(data.x)
        v_clean = torch.zeros(s.size(0), config.HIDDEN_DIM, 3, device=device)
        v_rot = torch.zeros(s.size(0), config.HIDDEN_DIM, 3, device=device)
        rel_pos_clean = data.pos[data.edge_index[1]] - data.pos[data.edge_index[0]]
        rel_pos_rot = pos_rot[data.edge_index[1]] - pos_rot[data.edge_index[0]]
        
        with torch.no_grad():
            for i in range(config.NUM_GNN_LAYERS):
                s_temp, v_temp = net.encoder.list_message[i](s, v_clean, data.edge_index, rel_pos_clean)
                s_c, v_c = s_temp + s, v_temp + v_clean
                s_temp, v_temp = net.encoder.list_update[i](s_c, v_c)
                s_c, v_c = s_temp + s_c, v_temp + v_c
                
                s_temp_r, v_temp_r = net.encoder.list_message[i](s, v_rot, data.edge_index, rel_pos_rot)
                s_r, v_r = s_temp_r + s, v_temp_r + v_rot
                s_temp_r, v_temp_r = net.encoder.list_update[i](s_r, v_r)
                s_r, v_r = s_temp_r + s_r, v_temp_r + v_r
                
            v_c_rotated = torch.matmul(v_c, R.t())
            equiv_pass = torch.allclose(v_r, v_c_rotated, atol=1e-5, rtol=1e-5)
            
    if inv_pass and node_inv_pass and equiv_pass:
        return "Pass"
    else:
        return "Fail"

def run_overfit_test(model_name, sample_batch, device):
    config.ENCODER_TYPE = model_name
    net = model.PretrainModel().to(device)
    net.train()
    data = sample_batch.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    initial_loss = None
    final_loss = None
    for step in range(50):
        optimizer.zero_grad()
        dist_preds, _ = net(data)
        loss = torch.nn.functional.mse_loss(dist_preds, data.dist_target)
        loss.backward()
        optimizer.step()
        if step == 0:
            initial_loss = loss.item()
        final_loss = loss.item()
        
    if final_loss < 0.1 or final_loss < 0.5 * initial_loss:
        return "Pass"
    else:
        return "Fail"

def run_training_and_evaluate(model_name, split_data, device):
    config.ENCODER_TYPE = model_name
    joint_transform = JointTransform()
    train_dataset = MoleculeGraphDataset(transform=joint_transform)
    train_dataset.files = split_data['train']
    val_dataset = MoleculeGraphDataset(transform=joint_transform)
    val_dataset.files = split_data['val']
    test_dataset = MoleculeGraphDataset(transform=joint_transform)
    test_dataset.files = split_data['test']
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)
    
    net = model.PretrainModel().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    
    best_val_rmse = float('inf')
    epoch_times = []
    start_time = time.time()
    process = psutil.Process()
    peak_memory = 0
    
    for epoch in range(20):
        epoch_start = time.time()
        net.train()
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            dist_preds, _ = net(data)
            loss = torch.nn.functional.mse_loss(dist_preds, data.dist_target)
            loss.backward()
            optimizer.step()
            
        net.eval()
        total_sq_err = 0.0
        total_edges = 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                dist_preds, _ = net(data)
                total_sq_err += ((dist_preds - data.dist_target) ** 2).sum().item()
                total_edges += data.dist_target.shape[0]
                
        val_rmse = np.sqrt(total_sq_err / total_edges) if total_edges > 0 else 0.0
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            
        epoch_times.append(time.time() - epoch_start)
        mem = process.memory_info().rss / (1024 * 1024)
        if mem > peak_memory:
            peak_memory = mem
            
    total_time = time.time() - start_time
    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    
    net.eval()
    test_embeddings = []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            _, mol_embedding = net.encoder(data.x, data.pos, data.edge_index, data.edge_attr, data.batch)
            test_embeddings.append(mol_embedding)
            
    if test_embeddings:
        test_embeddings = torch.cat(test_embeddings, dim=0)
        norm_embeddings = test_embeddings / (test_embeddings.norm(dim=-1, keepdim=True) + 1e-8)
        sim_matrix = torch.matmul(norm_embeddings, norm_embeddings.t())
        num_graphs = test_embeddings.shape[0]
        if num_graphs > 1:
            triu_indices = torch.triu_indices(num_graphs, num_graphs, offset=1)
            mean_similarity = sim_matrix[triu_indices[0], triu_indices[1]].mean().item()
        else:
            mean_similarity = 1.0
    else:
        mean_similarity = 0.0
        
    return {
        "rmse": best_val_rmse,
        "similarity": mean_similarity,
        "peak_mem": peak_memory,
        "avg_epoch_time": avg_epoch_time,
        "total_time": total_time
    }

def main():
    device = torch.device(config.DEVICE)
    print(f"Running smoke test validation suite on {device}...")
    
    split_path = os.path.join(config.PROCESSED_DIR, "smoke_split.pt")
    if not os.path.exists(split_path):
        print(f"Error: Split file not found at {split_path}")
        sys.exit(1)
    split_data = torch.load(split_path, weights_only=False)
    
    joint_transform = JointTransform()
    temp_dataset = MoleculeGraphDataset(transform=joint_transform)
    temp_dataset.files = split_data['train']
    sample_batch = next(iter(DataLoader(temp_dataset, batch_size=4, shuffle=False, num_workers=0)))
    
    architectures = ["schnet", "egnn", "painn", "dimenet", "mace"]
    results = {}
    
    for arch in architectures:
        print(f"\n--- Testing Architecture: {arch.upper()} ---")
        try:
            shape_res = run_shape_test(arch, sample_batch, device)
            print(f"Shape test: {shape_res}")
        except Exception as e:
            shape_res = "Fail"
            print(f"Shape test failed: {e}")
            
        try:
            equiv_res = run_equivariance_test(arch, sample_batch, device)
            print(f"Equivariance test: {equiv_res}")
        except Exception as e:
            equiv_res = "Fail"
            print(f"Equivariance test failed: {e}")
            
        try:
            overfit_res = run_overfit_test(arch, sample_batch, device)
            print(f"Overfit test: {overfit_res}")
        except Exception as e:
            overfit_res = "Fail"
            print(f"Overfit test failed: {e}")
            
        if shape_res == "Pass" and equiv_res == "Pass" and overfit_res == "Pass":
            e2e_res = "Yes"
        else:
            e2e_res = "No"
            
        print("Running short training run (20 epochs)...")
        try:
            metrics = run_training_and_evaluate(arch, split_data, device)
            print("Training completed successfully.")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Training failed: {e}")
            metrics = {
                "rmse": 0.0,
                "similarity": 0.0,
                "peak_mem": 0.0,
                "avg_epoch_time": 0.0,
                "total_time": 0.0
            }
            e2e_res = "No"
            
        results[arch] = {
            "shape": shape_res,
            "equiv": equiv_res,
            "overfit": overfit_res,
            "e2e": e2e_res,
            "rmse": metrics["rmse"],
            "similarity": metrics["similarity"],
            "peak_mem": metrics["peak_mem"],
            "avg_epoch_time": metrics["avg_epoch_time"],
            "total_time": metrics["total_time"]
        }
        
    print("\n\n==================== COMPARISON TABLE ====================")
    header = "| Architecture | Implementation Source | Shape Test | Equivariance Test | Overfit Test | E2E No Crash | Best Val RMSE | Mean Cosine Sim | Peak Memory (MB) | Time/Epoch (s) | Total Time (s) | Batch Size |"
    sep = "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    print(header)
    print(sep)
    
    sources = {
        "schnet": "Custom model.py (baseline)",
        "egnn": "egnn-pytorch (lucidrains)",
        "painn": "Custom model.py (self-contained)",
        "dimenet": "DimeNetPlusPlus (PyG wrapped)",
        "mace": "mace-torch (official)"
    }
    
    for arch in architectures:
        res = results[arch]
        source = sources[arch]
        line = f"| {arch.upper()} | {source} | {res['shape']} | {res['equiv']} | {res['overfit']} | {res['e2e']} | {res['rmse']:.4f} | {res['similarity']:.4f} | {res['peak_mem']:.1f} | {res['avg_epoch_time']:.2f} | {res['total_time']:.2f} | {config.BATCH_SIZE} |"
        print(line)
        
if __name__ == "__main__":
    main()
