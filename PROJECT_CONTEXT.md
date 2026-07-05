# Project Context — Multimodal 3D Peptide Embeddings

> Context file for coding agents. Read this fully before writing or editing code.
> It defines the goal, the architecture, the data, every design decision, and the
> build order. When in doubt, follow the decisions recorded here rather than
> defaulting to generic practice.

---

## 1. Goal

Build a **general-purpose embedding model for small peptides**, including
**non-canonical / modified residues** (phospho-residues, macrocycles,
N-methylation, D-amino acids, etc.).

- Output: a trained **encoder** that maps any peptide → a fixed-length embedding
  vector. The embedding is the deliverable; downstream tasks (e.g. MHC-I binding,
  AMP prediction) attach a small classifier on top of frozen/fine-tuned embeddings.
- Training paradigm: **self-supervised pretraining** on a large corpus, then
  **fine-tuning** on a small experimental non-canonical set.

This is a supervised research project (university). Architectural choices should
stay defensible and incremental; SchNet is a *baseline*, not a final commitment.

---

## 2. Architecture (high level)

Two-phase, two-branch multimodal design. **Build Phase 1 fully before Phase 2.**

```
                         ┌─ 3D GNN branch (SchNet) ──┐
peptide → SMILES/coords →┤                            ├─ late fusion → embedding
                         └─ SMILES Transformer branch ┘
        pretrain (ESMAtlas + PubChem) → fine-tune (RCSB non-canonical)
```

### Phase 1 — 3D GNN branch (CURRENT FOCUS)
- Atom-level molecular graph with **3D geometry**.
- Encoder: **SchNet** (distance-based message passing). Baseline; may later be
  compared against DimeNet / PaiNN / MACE.
- Self-supervised objective: **masked-atom prediction** (+ optional descriptor head).
- This branch alone is a complete, defensible project and the fallback deliverable.

### Phase 2 — SMILES branch + fusion (LATER, do not start until Phase 1 runs)
- Add a **SMILES Transformer** branch.
- **Late fusion** first (concatenate the two embeddings), measure 3D-only vs
  3D+SMILES. Only escalate to cross-attention / shared-Transformer fusion if late
  fusion proves the modalities are worth combining.
- Novel contribution: **residue-level cross-modal alignment** (align the SMILES
  view and the 3D view of each *residue*; a peptide-specific version of UniMAP's
  fragment-level alignment).

---

## 3. Key design decisions (and the reasoning — do not silently override)

| Decision | Choice | Why |
|---|---|---|
| Graph granularity | **Atom-level** (not residue-level) | A modified residue is mostly familiar atoms + a few new ones, so canonical→non-canonical transfer works naturally. No special residue vocabulary needed. |
| Dimensionality | **3D** (coordinates used) | Want geometry, not just connectivity. Caveat below. |
| 3D GNN model | **SchNet** to start | Distance-only → robust to imperfect/flexible-peptide geometry. Simplest stable baseline. Climb ladder (DimeNet→PaiNN→MACE) later as an experiment. |
| Edges | Drawn between atoms **within a distance cutoff (~5 Å)**, not just bonds | Captures through-space contacts (folding). This is what makes it a 3D graph. |
| Edge features | **RBF-expanded distance** + **bond-type one-hot** (incl. a "none" slot for non-bonded spatial neighbors) | Distance is continuous → RBF (smooth bumps), not one-hot. Bond type distinguishes real bonds from spatial contacts. |
| Node features | Chemical only, **no coordinates in node vector** | Raw xyz is not rotation/translation invariant. Geometry enters via edge distances only. |
| Categorical encoding | **one-hot** (with catch-all "other" slot) | Lets each category have its own learnable weights; avoids false ordering. |
| Continuous encoding | **RBF** for distance; normalize any other continuous feature | One-hot can't represent continuous values. |
| Long-range handling | **Virtual node** (+ residual connections) | Peptides are long; virtual node gives 2-hop global shortcut without stacking many layers (avoids over-smoothing). Distance-cutoff edges also bring folded-distant atoms close. |
| GNN depth | ~3–5 layers (default 4) | Captures functional groups / rings without over-smoothing. Tunable. |
| Pooling | mean to start; consider **jumping-knowledge** (combine all layers) | JK is a cheap win and pairs with the finding that intermediate layers carry useful signal. |
| Pretraining objective | **masked-atom prediction** (+ optional descriptor-prediction auxiliary head) | Graph analogue of PeptideCLM-2's masked-language-modeling; descriptor head helps smaller models. |
| Fusion (Phase 2) | **Late fusion (concatenate)** first | Robust, debuggable, enables clean 3D-only vs fused ablation. |

### The central caveat about 3D (keep in mind everywhere)
Reliable 3D for *modified, flexible* peptides barely exists. Experimental
structures are scarce (~hundreds in RCSB); predicted structures (ESMFold) are
canonical-only; generated conformers (RDKit/mETKDG) are guesses and a flexible
peptide has no single true shape. Therefore:
- Pretrain on data where 3D is reliable-enough and large (ESMAtlas predicted +
  PubChem precomputed).
- Fine-tune on scarce experimental non-canonical structures (RCSB).
- Treat "does added geometric sophistication help, or overfit noisy conformers?"
  as an open research question (SchNet vs heavier models is an experiment, not
  an assumption).

---

## 4. Data

### Currently on hand
- **Proteins/peptides**: `.pdb` files (from ESMAtlas; predicted structures, carry
  3D coordinates; per-residue **pLDDT lives in the B-factor column**).
- **Small molecules**: `.sdf` files (from PubChem; often carry precomputed 3D).

### Roles
- **PubChem (small molecules)** = broad chemical grounding (diverse functional
  groups, atoms, rings). NOT the non-canonical teacher.
- **ESMAtlas (canonical peptides)** = canonical peptide structure at scale.
  ESMFold is canonical-only → no non-canonical residues here.
- **RCSB (later, fine-tuning)** = scarce *experimental* non-canonical structures.
  Benchmark/fine-tuning scale only (~hundreds), not pretraining.
- The atom-level graph + RCSB fine-tuning is what adapts the model to
  non-canonical chemistry — not PubChem.

### Filters to apply (see `filters.py`)
- Elements: keep {C, N, O, S, P, H} + halogens {F, Cl, Br, I}; reject metals/exotic.
  (Halogens appear in PubChem, essentially never in peptides — element vocabulary
  must include halogen slots + an "other" catch-all so the source mismatch is harmless.)
- Size: heavy-atom count within bounds (drop tiny fragments and oversized molecules).
- Single covalent unit (no salts/mixtures; reject SMILES with ".").
- RDKit-parseable, has coordinates.
- ESMAtlas: **pLDDT cutoff** (threshold is an OPEN QUESTION flagged for the
  professor — keep it a config value, easy to change; it trades corpus size vs
  structure quality).
- ESMAtlas: filter to **short** sequences (peptide length), not full proteins.

### Loading
- `.pdb` → RDKit PDB reader → molecule + coords (+ pLDDT from B-factors).
- `.sdf` → RDKit SDF reader → molecule + coords (handle multi-molecule files).
- Sequence-only modified peptides (if ever needed) → **p2smi** → SMILES → RDKit.
  (Not needed for the PDB/SDF data already on hand.)
- For molecules lacking coordinates: generate with **RDKit ETKDG**, and
  **mETKDG** (`useMacrocycleTorsions=True`) for cyclic/macrocyclic peptides.
  Expect a non-trivial macrocycle failure rate; track and log failures.

---

## 5. Codebase structure

```
peptide-embeddings/
├── config.py              # all settings/hyperparameters in one place
├── data/
│   ├── raw/               # downloaded .pdb and .sdf files
│   └── processed/         # saved graph objects (build once, reuse)
├── src/
│   ├── loaders.py         # PDB/SDF → (rdkit_mol, coords, meta)
│   ├── filters.py         # keep/discard decisions, log rejection reasons
│   ├── features.py        # atom & edge feature encoding, dims
│   ├── graph_builder.py   # (mol, coords) → PyG Data   [LINCHPIN]
│   ├── dataset.py         # PyG Dataset + masking transform
│   ├── model.py           # SchNet encoder + pretraining heads
│   ├── pretrain.py        # masked-atom training loop
│   └── utils.py           # logging, checkpointing, helpers
├── scripts/
│   ├── 00_inspect.py      # inspect raw data BEFORE writing parsers
│   ├── 01_build_graphs.py # loaders→filters→graph_builder, save to processed/
│   └── 02_pretrain.py     # run training
└── tests/
    └── test_graph.py      # verify graph builder by hand on a few molecules
```

---

## 6. File-by-file responsibilities

### `config.py` (write first)
All paths, filter thresholds (pLDDT, size bounds, allowed elements), graph
settings (distance cutoff ~5 Å, #RBF centers ~16, RBF range), model settings
(hidden dim, #layers, embedding size), training settings (lr, batch size,
mask rate ~15%, epochs). Everything else imports from here.

### `scripts/00_inspect.py` (run before parsing)
Open a few `.pdb` and `.sdf` files; print atom counts, sequence lengths,
pLDDT (B-factor) values, element types, coordinate presence, single-vs-multi
component. Print size/pLDDT distributions over a larger sample. Purpose: confirm
the data is what we assume before writing loaders/filters.

### `src/loaders.py`
- `load_pdb(filepath)` → (mol, coords, meta{pLDDT,...}); None on parse failure.
- `load_sdf(filepath)` → iterate molecules → (mol, coords, meta).
- `standardize(mol)` → strip salts/solvent, neutralize, fixed H policy
  (decide explicit vs implicit Hs ONCE, apply everywhere).
- `load_any(filepath)` → dispatch by extension.
- Contract: every loader returns the same `(mol, coords, meta)` shape.

### `src/filters.py`
- `passes_element_filter`, `passes_size_filter`, `passes_single_component`,
  `passes_plddt`, `passes_parseable`.
- `apply_all_filters(mol, meta)` → (keep: bool, reasons: list).
- Log counts per rejection reason (thresholds will be tuned).

### `src/features.py`
- `onehot(value, choices)` → one-hot list with catch-all "other" slot.
- `atom_features(atom)` → node vector: element, degree, totalHs, formal charge,
  hybridization, aromaticity, in-ring, chirality. Fixed length.
- `rbf_expand(distance, centers, width)` → smooth bumps for a distance.
- `edge_features(distance, bond_or_none)` → RBF distance + bond-type one-hot
  (with "none" slot).
- Expose `NODE_FEATURE_DIM`, `EDGE_FEATURE_DIM` (computed once; model reads these).

### `src/graph_builder.py` (LINCHPIN — most care here)
- `build_graph(mol, coords)`:
  1. node matrix `x` = stack of `atom_features`.
  2. positions array from coords.
  3. all-pairs (or KD-tree/cell-list for big mols) distance; if < cutoff, add
     edge (both directions) + `edge_features`.
  4. mark each in-cutoff pair as real-bond (correct bond-type slot) vs "none".
  5. return PyG `Data(x, edge_index, edge_attr, pos)`.
  6. return None on degenerate graphs (single atom / no edges).
- Performance: all-pairs is O(atoms²); use neighbor search for large molecules.

### `tests/test_graph.py` (run before trusting builder)
On 3–5 deliberately chosen molecules (tiny / cyclic / sulfur-containing):
check node count == atom count; edge count sane (not 0, not fully connected);
feature dims match the constants; all edge distances < cutoff; a known bond has
the right bond-type slot.

### `scripts/01_build_graphs.py`
Iterate raw files → `load_any` → `apply_all_filters` → `build_graph`; save graphs
to `data/processed/`. Log loaded / passed-each-filter / built / failed.
**Run on ~10 stress-test molecules first, confirm, then scale.**

### `src/dataset.py`
- PyG `Dataset`/`InMemoryDataset` loading processed graphs.
- **Masking transform**: per batch, hide ~15% of atoms' features, record which
  were masked + their true values (this creates the self-supervised target).

### `src/model.py`
- `Encoder`: input embedding (node features → hidden dim) → SchNet layers
  (PyG SchNet / interaction blocks) → pooling. Optional jumping-knowledge.
  Produces the molecule embedding. KEEP SEPARABLE from heads.
- `MaskedAtomHead`: per-atom vectors → predict masked atom identities.
- `DescriptorHead` (optional): pooled embedding → predict RDKit descriptors.
- `PretrainModel`: wraps encoder + heads.
- After pretraining: discard heads, keep `Encoder` = the deliverable.

### `src/pretrain.py` + `scripts/02_pretrain.py`
- `compute_loss` (masked-atom cross-entropy + optional descriptor MSE).
- `train_one_epoch`, `evaluate`, `train(config)` with encoder checkpointing.
- Script wires config → dataset → model → train; saves encoder.
- **Run on small subset first** (loss computes & decreases), then scale.

---

## 7. Build order (checklist)

```
0.  config.py
0.5 scripts/00_inspect.py   ← look at raw data, confirm format
1.  src/loaders.py          ← PDB/SDF → (mol, coords, meta)
2.  src/filters.py          ← keep/discard
3.  src/features.py         ← atom & edge encoding
4.  src/graph_builder.py    ← (mol,coords) → graph     [LINCHPIN]
5.  tests/test_graph.py     ← verify builder by hand
6.  scripts/01_build_graphs.py ← run on ~10, then scale, save graphs
7.  src/dataset.py          ← PyG dataset + masking transform
8.  src/model.py            ← SchNet encoder + masked-atom head
9.  src/pretrain.py / 02    ← training loop, run on subset, then scale
```

Phase 2 (later, only after Phase 1 runs end to end): add `src/smiles_encoder.py`
(SMILES Transformer), `src/fusion.py` (late fusion), extend the dataset to also
yield SMILES, add cross-modal + residue-level alignment objectives.

---

## 8. Engineering disciplines (enforce these)

1. **Inspect before parsing** — confirm data reality first (step 0.5).
2. **Verify the graph builder by hand** (step 5) — silent graph bugs train wrong
   models that still "look" fine.
3. **Small subset before scale** at steps 6 and 9 — prove it runs on ~10 before
   millions. The first test set should be DELIBERATE and stress-testing (varied
   sizes, ≥1 cyclic, sulfur-containing, a couple of PubChem small molecules), not
   random — random hides edge cases.
4. **Save processed graphs** — build once, reuse; never rebuild every run.
5. **Keep encoder separable from heads** — so the deliverable extracts cleanly.
6. **Log filter rejection reasons** — thresholds (esp. pLDDT) will be tuned.
7. **No coordinates in node features** — geometry only via edge distances.
8. **Categorical → one-hot (+other); continuous → RBF/normalized.**
9. **Don't start Phase 2 until Phase 1 trains end to end.**

---

## 9. Training paradigm summary

- **Pretrain** (large: ESMAtlas + PubChem, canonical/small-molecule, no labels):
  masked-atom prediction (+ optional descriptor head). Random init, normal LR.
- **Fine-tune** (small: RCSB experimental non-canonical): continue from pretrained
  weights, **low LR, encoder unfrozen** (the chemistry shift is large enough that
  frozen features won't capture it; consider gradual unfreezing — head first, then
  encoder). Watch for catastrophic forgetting and overfitting the tiny set; hold
  out a clean test split.
- **Deliverable**: the fine-tuned encoder (checkpoint = architecture + weights).
  Feed any peptide → embedding.

---

## 10. Reference points (for orientation, not blueprints)

- **PeptideCLM-2**: corpus recipe (PubChem + ESMAtlas + LIPID MAPS via p2smi),
  masked-language-modeling, descriptor-prediction trick. It is SMILES+Transformer,
  NOT a GNN, NOT 3D — we borrow its *data recipe and objectives*, not its model.
- **UniMAP**: 2D SMILES+graph fusion via shared Transformer, fragment-level
  alignment. We borrow the *fragment idea* (→ residues), not the 2D architecture.
- **MolGraph-xLSTM**: supervised 2D GNN; source of borrowable tricks — **virtual
  node**, **jumping knowledge**, **supervised contrastive loss** (for fine-tuning).
- **Layer-wise probing study**: probes existing 3D force-field models (MACE,
  Pos-EGNN, Orb, Uni-Mol); finding that **intermediate layers** can give better
  representations than the last — motivates jumping-knowledge / layer selection.
- **MMGX**: 2D atom-graph + attention; attention-based interpretability idea to
  borrow later.

SchNet is NOT taken from any single uploaded paper; it's the standard distance-based
3D baseline informed by the general 3D-GNN literature.

---

## 11. Work Accomplished and Codebase Inventory (Phase 1 Status)

Phase 1 (3D GNN branch) setup, encoder architectures, pretraining framework, and frozen-representation evaluations have been successfully completed. 

### Files Inventory and Roles
- **[config.py](file:///c:/Users/Dhanush/Desktop/project/config.py)**: Central hub for all settings, directories, vocabulary, features, model architectures, and pretraining hyperparameters.
- **[model.py](file:///c:/Users/Dhanush/Desktop/project/model.py)**: Holds GNN Encoders and Pretraining Heads. Includes:
  - `Encoder` (SchNet GNN): Distance-based interaction block utilizing radial basis functions (RBF).
  - `EGNNEncoder`: Wrapper around `egnn_pytorch` sparse implementation.
  - `PaiNNEncoder`: Equivariant Message Passing implementation (using `lin1` and `lin2` separate linear projection layers).
  - `DimeNetPlusPlusEncoder`: Wraps PyG's DimeNet++ with custom triplet angle computation.
  - `MACEEncoder`: Wrapper around standard MACE library using atomic number mappings (incorporating 0 as a sentinel index).
  - `DistanceDenoisingHead` & `DescriptorHead`: Self-supervised pretraining heads.
- **[scripts/qm9_probe.py](file:///c:/Users/Dhanush/Desktop/project/scripts/qm9_probe.py)**: Performs frozen-embedding evaluation of the pretrained SchNet encoder (using `encoder_epoch_40.pt` checkpoint) on the QM9 dataset targeting the HOMO-LUMO gap. Features:
  - Robust, automated raw files mirror/download overrides to bypass Figshare WAF block.
  - Unbiased Murcko scaffold-splitting mapping whole groups randomly to splits.
  - Fully formatted ASCII outputs for compatibility across Windows console encodings.
- **[scripts/explore_res.py](file:///c:/Users/Dhanush/Desktop/project/scripts/explore_res.py)**: Handles download, extraction, and statistics parsing for ATOM3D's Residue Identity (RES) dataset.
- **[scripts/fast_stats.py](file:///c:/Users/Dhanush/Desktop/project/scripts/fast_stats.py)**: Efficiently computes sample statistics for the extracted local RES dataset and automatically deletes the large 4.1 GB zip archive.

### QM9 Evaluation Results (HOMO-LUMO Gap Prediction)
The pretrained SchNet encoder was evaluated by freezing its embeddings and training linear (Ridge) and non-linear (2-layer MLP) probes. Baseline comparisons were performed against a Mean Predictor (floor) and Morgan Fingerprints:

| Model / Method | MAE (eV) | RMSE (eV) | Notes |
|:---|:---:|:---:|:---|
| **Mean Predictor** (floor) | 1.0858 | 1.2771 | Predicts training mean |
| **Morgan Fingerprints** + Ridge | 0.4412 | 0.5842 | ECFP4 baseline |
| **Pretrained SchNet** + Ridge | 0.5570 | 0.7123 | Frozen encoder embeddings |
| **Pretrained SchNet** + MLP | 0.7815 | 0.9370 | Full-batch optimization |

