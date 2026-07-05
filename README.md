# 3D Molecular & Peptide Embeddings

**Status: Ongoing** · Phase 1 complete (probing) · Phase 2 in design

General-purpose 3D structural embeddings for small molecules and short peptides, learned by self-supervised pretraining of 3D graph neural network encoders. The long-term goal is a multimodal model that aligns 3D structure with the SMILES sequence at the **residue level** — a cross-modal signal not covered by existing sequence-only work such as PeptideCLM.

The project runs in two phases: (1) pretrain and benchmark a set of 3D GNN encoders, and (2) extend the best encoder into a multimodal architecture with residue-level cross-modal alignment.

---

## Roadmap

### Phase 1 — 3D GNN pretraining ✅ (probing done, convergence runs pending)

- [x] Implement five 3D GNN architectures behind a shared, config-selectable interface
- [x] Build the data pipeline (PubChem small molecules, AlphaFold DB peptides)
- [x] Move radius-graph edge construction to load-time computation (fixed a disk/storage bottleneck)
- [x] Distance-denoising pretraining objective
- [x] Linear-probe comparison across all five encoders
- [ ] Re-run all encoders to full convergence on HPC (blocked on cluster access)

### Phase 2 — multimodal cross-modal alignment 🔜 (design stage)

- [x] Conceptual architecture defined (3D GNN branch + SMILES Transformer branch)
- [ ] Resolve alignment-mechanism design questions (see below)
- [ ] Implement residue-level pooling on both branches
- [ ] Implement alignment loss + projection heads
- [ ] Downstream benchmark evaluation

---

## Phase 1 results

Linear-probing results across the five architectures, measured as error reduction over a mean-prediction floor. All models were trained on Google Colab and **none reached full convergence** — these use best-epoch checkpoints under compute limits, so the ordering is indicative rather than final.

| Architecture | Error reduction vs. mean floor | Type |
|---|---|---|
| MACE | ~70% (leading) | Equivariant |
| PaiNN | ~66% | Equivariant |
| EGNN | ~60% | Equivariant |
| DimeNet++ | ~55% | Invariant (directional) |
| SchNet | ~49% | Invariant |

Full-convergence runs on the HPC cluster are required before these numbers are treated as the final Phase 1 comparison.

---

## Architecture

### Phase 1

- **Encoders:** SchNet, EGNN, PaiNN, DimeNet++, MACE — all selectable via a config string behind one shared interface.
- **Pretraining objective:** distance denoising — predicting clean interatomic distance residuals per edge. This is an invariant target, chosen so it stays compatible with invariant encoders like SchNet.
- **Data:** small molecules from PubChem (with precomputed 3D coordinates); short peptides (5–50 residues) from AlphaFold DB, filtered by pLDDT and clustered with MMseqs2.

### Phase 2 (planned)

Two branches over the same molecule:

- **3D GNN branch** — atom-level embeddings, carried over from Phase 1.
- **SMILES Transformer branch** — token-level embeddings of the SMILES string.

Both branches are sub-residue (atoms vs. tokens), so each is **pooled up to residue vectors**, projected into a shared space, L2-normalized, and matched. Residue-level alignment is the core research contribution.

**Open design questions:**
- Alignment loss: contrastive (InfoNCE) vs. token-level correspondence (regression). Contrastive is the current lead — discriminative and needs no anti-collapse machinery. This is flagged as the piece needing the most design work.
- Residue-to-SMILES-span mapping (which tokens make up residue *i*).
- Residue-membership bookkeeping on the 3D side (which atoms make up residue *i*), which must be preserved through radius-graph construction.
- Whether the 3D branch stays frozen or is fine-tuned during multimodal training. Freezing first is the lower-risk starting point.

---

## Evaluation

Benchmarks must reflect the pretraining domain — domain mismatch is treated as disqualifying.

**In use / shortlisted:**
- Atom3D: RES (subsampled within CATH-topology split partitions to avoid leakage), SMP, MSP (optional transfer-gap probe)
- Megascale / Tsuboyama stability dataset
- TDC ADMET tasks (via `pytdc`)

**Ruled out:** LBA, LEP, PIP — the encoders never modeled protein–ligand complexes, so these are out-of-domain.

---

## Key learnings

- **Denoising target must match encoder symmetry.** Invariant encoders (e.g. SchNet) can't predict directional vectors, so coordinate denoising plateaus. Distance denoising is the correct invariant target.
- **Edge-distance leakage.** Masked-atom prediction can reach near-perfect accuracy for the wrong reason — bond lengths encode element identity. A failure mode to watch for.
- **Equivariant models unlock richer pretraining.** Equivariant architectures (PaiNN, MACE) support coordinate denoising + masked-atom prediction combined; invariant models can't.
- **Storage vs. compute tradeoff.** Materializing full radius-graph edges to disk was a bottleneck; load-time computation via PyG's `radius_graph` fixed it.
- **Subsampling within split partitions.** For datasets like RES, subsampling must happen inside the CATH-topology split partitions to prevent leakage.
- **Domain match is non-negotiable** for benchmark selection (see Evaluation).

---

## Stack

- **Framework:** PyTorch Geometric (PyG)
- **Compute:** Google Colab (current), IITD HPC cluster (pending access)
- **Data sources:** PubChem, AlphaFold DB
- **Benchmarks:** Atom3D, Megascale/Tsuboyama, TDC (`pytdc`)

---

## Current blockers

- HPC cluster access pending — required for full-convergence Phase 1 runs before Phase 2 implementation begins.
