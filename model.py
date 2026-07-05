import torch
import torch.nn as nn
import numpy as np
from torch_geometric.nn import MessagePassing, global_mean_pool, global_add_pool, GlobalAttention
from torch_geometric.nn.models import DimeNetPlusPlus
import config

class SchNetInteractionBlock(MessagePassing):
    def __init__(self, hidden_dim, edge_dim):
        super().__init__(aggr='add')
        self.mlp = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.lin = nn.Linear(hidden_dim, hidden_dim)
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x, edge_index, edge_attr):
        edge_weight = self.mlp(edge_attr)
        x_lin = self.lin(x)
        m = self.propagate(edge_index, x=x_lin, edge_weight=edge_weight)
        return x + self.update_mlp(m)

    def message(self, x_j, edge_weight):
        return x_j * edge_weight

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.node_emb = nn.Embedding(len(config.ALLOWED_ELEMENTS) + 1, config.HIDDEN_DIM)
        self.layers = nn.ModuleList([
            SchNetInteractionBlock(config.HIDDEN_DIM, config.RBF_NUM_CENTERS)
            for _ in range(config.NUM_GNN_LAYERS)
        ])
        self.use_jk = config.USE_JUMPING_KNOWLEDGE
        if self.use_jk:
            self.jk_lin = nn.Linear(config.HIDDEN_DIM * config.NUM_GNN_LAYERS, config.HIDDEN_DIM)
        if config.POOLING == "mean":
            self.pool = global_mean_pool
        elif config.POOLING == "sum":
            self.pool = global_add_pool
        elif config.POOLING == "attention":
            gate_nn = nn.Sequential(nn.Linear(config.HIDDEN_DIM, 1))
            self.pool = GlobalAttention(gate_nn)
        else:
            self.pool = global_mean_pool

    def forward(self, x, pos, edge_index, edge_attr, batch):
        x = self.node_emb(x)
        layer_outputs = []
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
            layer_outputs.append(x)
        if self.use_jk:
            x_out = torch.cat(layer_outputs, dim=-1)
            x_out = self.jk_lin(x_out)
        else:
            x_out = layer_outputs[-1]
        atom_vectors = x_out
        pooled = self.pool(atom_vectors, batch)
        return atom_vectors, pooled

class EGNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        from egnn_pytorch import EGNN_Sparse
        self.node_emb = nn.Embedding(len(config.ALLOWED_ELEMENTS) + 1, config.HIDDEN_DIM)
        self.layers = nn.ModuleList([
            EGNN_Sparse(feats_dim=config.HIDDEN_DIM, m_dim=config.HIDDEN_DIM, update_coors=True)
            for _ in range(config.NUM_GNN_LAYERS)
        ])

    def forward(self, x, pos, edge_index, edge_attr, batch):
        feats = self.node_emb(x)
        inp = torch.cat([pos, feats], dim=-1)
        for layer in self.layers:
            inp = layer(inp, edge_index, batch=batch)
        node_feats = inp[:, 3:]
        return node_feats, global_mean_pool(node_feats, batch)

class CosineCutoff(nn.Module):
    def __init__(self, cutoff=5.0):
        super().__init__()
        self.cutoff = cutoff

    def forward(self, distances):
        cutoffs = 0.5 * (torch.cos(distances * 3.141592653589793 / self.cutoff) + 1.0)
        cutoffs = cutoffs * (distances < self.cutoff).float()
        return cutoffs

class BesselBasis(nn.Module):
    def __init__(self, cutoff=5.0, n_rbf=20):
        super().__init__()
        self.cutoff = cutoff
        freqs = torch.arange(1, n_rbf + 1) * 3.141592653589793 / cutoff
        self.register_buffer("freqs", freqs)

    def forward(self, inputs):
        norm = torch.norm(inputs, p=2, dim=1)
        ax = torch.outer(norm, self.freqs)
        sinax = torch.sin(ax)
        norm_denom = torch.where(norm == 0, torch.tensor(1.0, device=norm.device), norm)
        return sinax / norm_denom.unsqueeze(1)

class MessagePassPaiNN(MessagePassing):
    def __init__(self, num_feat, out_channels, cut_off=5.0, n_rbf=20):
        super().__init__(aggr="add")
        self.lin1 = nn.Linear(num_feat, out_channels)
        self.lin2 = nn.Linear(out_channels, 3 * out_channels)
        self.lin_rbf = nn.Linear(n_rbf, 3 * out_channels)
        self.RBF = BesselBasis(cut_off, n_rbf)
        self.f_cut = CosineCutoff(cut_off)
        self.num_feat = num_feat

    def forward(self, s, v, edge_index, edge_attr):
        s_flat = s.flatten(-1)
        v_flat = v.flatten(-2)
        flat_shape_s = s_flat.shape[-1]
        flat_shape_v = v_flat.shape[-1]
        x = torch.cat([s_flat, v_flat], dim=-1)
        x = self.propagate(
            edge_index,
            x=x,
            edge_attr=edge_attr,
            flat_shape_s=flat_shape_s,
            flat_shape_v=flat_shape_v,
        )
        return x

    def message(self, x_j, edge_attr, flat_shape_s, flat_shape_v):
        s_j, v_j = torch.split(x_j, [flat_shape_s, flat_shape_v], dim=-1)
        rbf = self.RBF(edge_attr)
        ch1 = self.lin_rbf(rbf)
        cut = self.f_cut(torch.norm(edge_attr, p=2, dim=-1))
        W = ch1 * cut.unsqueeze(1)
        phi = self.lin2(torch.nn.functional.silu(self.lin1(s_j)))
        left, dsm, right = torch.split(phi * W, self.num_feat, dim=-1)
        normalized = torch.nn.functional.normalize(edge_attr, p=2, dim=-1)
        v_j = v_j.reshape(-1, self.num_feat, 3)
        hadamard_right = right.unsqueeze(-1) * normalized.unsqueeze(1)
        hadamard_left = v_j * left.unsqueeze(-1)
        dvm = hadamard_left + hadamard_right
        return torch.cat((dsm, dvm.flatten(-2)), dim=-1)

    def update(self, out_aggr, flat_shape_s, flat_shape_v):
        s_j, v_j = torch.split(out_aggr, [flat_shape_s, flat_shape_v], dim=-1)
        return s_j, v_j.reshape(-1, self.num_feat, 3)

class UpdatePaiNN(nn.Module):
    def __init__(self, num_feat, out_channels):
        super().__init__()
        self.lin_up = nn.Linear(2 * num_feat, out_channels)
        self.denseU = nn.Linear(num_feat, out_channels, bias=False)
        self.denseV = nn.Linear(num_feat, out_channels, bias=False)
        self.lin2 = nn.Linear(out_channels, 3 * out_channels)
        self.num_feat = num_feat

    def forward(self, s, v):
        v_ut = torch.transpose(v, 1, 2)
        U = torch.transpose(self.denseU(v_ut), 1, 2)
        V = torch.transpose(self.denseV(v_ut), 1, 2)
        UV = torch.einsum("ijk,ijk->ij", U, V)
        nV = torch.norm(V, dim=-1)
        s_u = self.lin2(torch.nn.functional.silu(self.lin_up(torch.cat([s, nV], dim=-1))))
        top, middle, bottom = torch.split(s_u, self.num_feat, dim=-1)
        dvu = v * top.unsqueeze(-1)
        dsu = middle * UV + bottom
        return dsu, dvu

class PaiNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.node_emb = nn.Embedding(len(config.ALLOWED_ELEMENTS) + 1, config.HIDDEN_DIM)
        self.list_message = nn.ModuleList([
            MessagePassPaiNN(config.HIDDEN_DIM, config.HIDDEN_DIM, config.DISTANCE_CUTOFF, config.RBF_NUM_CENTERS)
            for _ in range(config.NUM_GNN_LAYERS)
        ])
        self.list_update = nn.ModuleList([
            UpdatePaiNN(config.HIDDEN_DIM, config.HIDDEN_DIM)
            for _ in range(config.NUM_GNN_LAYERS)
        ])
        self.lin1 = nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM)
        self.lin2 = nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM)

    def forward(self, x, pos, edge_index, edge_attr, batch):
        s = self.node_emb(x)
        v = torch.zeros(s.size(0), config.HIDDEN_DIM, 3, device=s.device)
        rel_pos = pos[edge_index[1]] - pos[edge_index[0]]
        for i in range(config.NUM_GNN_LAYERS):
            s_temp, v_temp = self.list_message[i](s, v, edge_index, rel_pos)
            s, v = s_temp + s, v_temp + v
            s_temp, v_temp = self.list_update[i](s, v)
            s, v = s_temp + s, v_temp + v
        s = self.lin2(torch.nn.functional.silu(self.lin1(s)))
        return s, global_mean_pool(s, batch)

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

class DimeNetPlusPlusWrapper(DimeNetPlusPlus):
    def forward(self, z, pos, edge_index, batch=None):
        i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji = compute_triplets(
            edge_index, num_nodes=z.size(0))
        dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()
        pos_jk, pos_ij = pos[idx_j] - pos[idx_k], pos[idx_i] - pos[idx_j]
        a = (pos_ij * pos_jk).sum(dim=-1)
        b = torch.cross(pos_ij, pos_jk, dim=1).norm(dim=-1)
        angle = torch.atan2(b, a)
        rbf = self.rbf(dist)
        sbf = self.sbf(dist, angle, idx_kj)
        x = self.emb(z, rbf, i, j)
        P = self.output_blocks[0](x, rbf, i, num_nodes=pos.size(0))
        for interaction_block, output_block in zip(self.interaction_blocks,
                                                   self.output_blocks[1:]):
            x = interaction_block(x, rbf, sbf, idx_kj, idx_ji)
            P = P + output_block(x, rbf, i, num_nodes=pos.size(0))
        return P

class DimeNetPlusPlusEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.dimenet = DimeNetPlusPlusWrapper(
            hidden_channels=config.HIDDEN_DIM,
            out_channels=config.HIDDEN_DIM,
            num_blocks=3,
            int_emb_size=64,
            basis_emb_size=8,
            out_emb_channels=config.HIDDEN_DIM,
            num_spherical=3,
            num_radial=6,
            cutoff=config.DISTANCE_CUTOFF
        )

    def forward(self, x, pos, edge_index, edge_attr, batch):
        atom_feats = self.dimenet(x, pos, edge_index, batch)
        return atom_feats, global_mean_pool(atom_feats, batch)

class MACEEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        import e3nn
        from mace.modules import MACE, RealAgnosticResidualInteractionBlock
        num_elements = len(config.ALLOWED_ELEMENTS) + 1
        atomic_numbers = [6, 7, 8, 16, 15, 1, 9, 17, 35, 53, 0][:num_elements]
        self.mace = MACE(
            r_max=config.DISTANCE_CUTOFF,
            num_bessel=8,
            num_polynomial_cutoff=5,
            max_ell=config.MACE_MAX_ELL,
            interaction_cls=RealAgnosticResidualInteractionBlock,
            interaction_cls_first=RealAgnosticResidualInteractionBlock,
            num_interactions=config.MACE_NUM_INTERACTIONS,
            num_elements=num_elements,
            hidden_irreps=e3nn.o3.Irreps(f"{config.HIDDEN_DIM}x0e + {config.HIDDEN_DIM}x1o"),
            MLP_irreps=e3nn.o3.Irreps("16x0e"),
            atomic_energies=np.zeros(num_elements),
            avg_num_neighbors=10.0,
            atomic_numbers=atomic_numbers,
            correlation=config.MACE_CORRELATION,
            gate=torch.nn.functional.silu
        )
        in_dim = 4 * config.HIDDEN_DIM * config.MACE_NUM_INTERACTIONS
        self.proj = nn.Linear(in_dim, config.HIDDEN_DIM)

    def forward(self, x, pos, edge_index, edge_attr, batch):
        num_graphs = int(batch.max().item() + 1)
        node_attrs = torch.nn.functional.one_hot(x, num_classes=len(config.ALLOWED_ELEMENTS) + 1).float()
        ptr = torch.cat([torch.tensor([0], device=batch.device), torch.cumsum(torch.bincount(batch), dim=0)])
        data_dict = {
            'positions': pos,
            'node_attrs': node_attrs,
            'edge_index': edge_index,
            'shifts': torch.zeros(edge_index.shape[1], 3, dtype=pos.dtype, device=pos.device),
            'cell': torch.zeros(num_graphs, 3, 3, dtype=pos.dtype, device=pos.device),
            'ptr': ptr,
            'batch': batch
        }
        out = self.mace(data_dict, training=self.training, compute_force=False)
        atom_feats = self.proj(out['node_feats'])
        return atom_feats, global_mean_pool(atom_feats, batch)

def get_encoder(encoder_name):
    name = encoder_name.lower()
    if name == "schnet":
        return Encoder()
    elif name == "egnn":
        return EGNNEncoder()
    elif name == "painn":
        return PaiNNEncoder()
    elif name == "dimenet":
        return DimeNetPlusPlusEncoder()
    elif name == "mace":
        return MACEEncoder()
    elif name == "smiles" or name.startswith("smiles_"):
        # SMILES Transformer branch (Phase 2). Lazy-imported so the 3D GNN path
        # does not require torchtune/transformers. Select the size preset via the
        # config string, e.g. "smiles" (small), "smiles_base", "smiles_large".
        from smiles_encoder import get_smiles_encoder
        size = name.split("_", 1)[1] if "_" in name else "small"
        return get_smiles_encoder(size)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_name}")

class DistanceDenoisingHead(nn.Module):
    def __init__(self):
        super().__init__()
        in_dim = config.HIDDEN_DIM * 2 + config.RBF_NUM_CENTERS
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, config.HIDDEN_DIM),
            nn.SiLU(),
            nn.Linear(config.HIDDEN_DIM, 1)
        )
        
    def forward(self, atom_vectors, edge_index, edge_attr):
        row, col = edge_index
        h_i = atom_vectors[row]
        h_j = atom_vectors[col]
        edge_rep = torch.cat([h_i + h_j, torch.abs(h_i - h_j), edge_attr], dim=-1)
        return self.mlp(edge_rep).squeeze(-1)

class DescriptorHead(nn.Module):
    def __init__(self):
        super().__init__()
        num_descriptors = len(config.DESCRIPTOR_LIST)
        self.mlp = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM),
            nn.SiLU(),
            nn.Linear(config.HIDDEN_DIM, num_descriptors)
        )
        
    def forward(self, mol_embedding):
        return self.mlp(mol_embedding)

class PretrainModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = get_encoder(config.ENCODER_TYPE)
        self.denoise_head = DistanceDenoisingHead()
        self.use_descriptor = config.USE_DESCRIPTOR_HEAD
        if self.use_descriptor:
            self.descriptor_head = DescriptorHead()
            
    def forward(self, data):
        batch = data.batch if hasattr(data, 'batch') and data.batch is not None else torch.zeros(data.x.shape[0], dtype=torch.long, device=data.x.device)
        atom_vectors, mol_embedding = self.encoder(data.x, data.pos, data.edge_index, data.edge_attr, batch)
        dist_preds = self.denoise_head(atom_vectors, data.edge_index, data.edge_attr)
        desc_preds = None
        if self.use_descriptor:
            desc_preds = self.descriptor_head(mol_embedding)
        return dist_preds, desc_preds
