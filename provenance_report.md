# Dataset Provenance Report
 
This report documents the collection, filtering, validation, and curation results for the Multimodal 3D Peptide Embeddings training dataset.
 
## Target Dataset Composition
- **Total Targets:** exactly 50,000 structures (25,000 peptides + 25,000 small molecules).
- **Target Ratio:** 1:1 balance enforced.
 
---
 
## 1. Peptides (AlphaFold DB)
- **Initial Candidates Streamed:** 500
- **Candidates Attempted/Fetched:** 0
- **Dropped at Filter Stages:**
  - Length filter drop (outside 5-50 AA): 0
  - Low-confidence prediction drop (pLDDT <= 70.0): 0
  - Deduplication drop (InChIKey overlaps): 0
- **Skipped for Lacking 3D Coordinates:** 0
- **Failed Validation Gate:** 0
- **Final Retained Peptides (after sequence clustering & balancing):** 125
- **Fraction of Usable 3D Coordinate Structures:** 0.00% (usable / attempted)
 
---
 
## 2. Small Molecules (PubChem)
- **CIDs Scanned (Property Level):** 0
- **CIDs Attempted/Fetched (3D SDF):** 0
- **Dropped at Filter Stages:**
  - Size out of bounds (MW or atom count): 0
  - SMILES length < 20: 0
  - Disconnected component splits: 0 (kept largest fragment)
  - Silicon/polymeric skips: 0
  - Deduplication drop (InChIKey overlaps): 0
- **Skipped for Lacking 3D Coordinates:** 0
- **Failed Validation Gate:** 0
- **Final Retained Small Molecules (after balancing):** 125
- **Fraction of Usable 3D Coordinate Structures:** 0.00% (usable / attempted)
 
---
 
## 3. Dataset Deduplication Summary
- Deduplication was executed using canonical InChIKeys.
- All InChIKeys were tracked globally to ensure no overlapping structures existed between the peptide and small-molecule pools.

*Report generated at 2026-07-03 00:49:10 (Local Time).*
