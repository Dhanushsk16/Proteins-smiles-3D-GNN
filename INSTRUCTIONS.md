# Project Setup & Dependency Guide

This guide is for any AI Agent or developer resuming work on a new device. It details the required Python libraries, installation order, and workarounds for system-specific compiler limitations.

---

## 1. Quick Install Script
To automate library installation on the new device, run:
```bash
# Core Machine Learning & Chemistry
pip install torch numpy pandas scipy scikit-learn rdkit requests

# PyTorch Geometric (PyG) and EGNN
pip install torch-geometric egnn-pytorch

# MACE Dependencies
pip install e3nn mace-torch

# ATOM3D Core dependencies (installed individually to isolate compiler issues)
pip install biopython click easy-parallel h5py lmdb msgpack pyrr python-dotenv tables

# Install ATOM3D without compilation dependencies (avoids Windows MSVC compilation errors for freesasa)
pip install atom3d --no-deps
```

---

## 2. Windows / Python 3.13+ Workarounds
On modern Python versions (e.g. 3.12, 3.13) under Windows, the package `freesasa` (a dependency of `atom3d`) fails to build due to missing C++ compilers. 

Because we only use `atom3d.datasets` for RES file loading, we can bypass this completely by mocking `freesasa` in scripts. Ensure the following snippet is at the top of any ATOM3D script (e.g., `explore_res.py`, `fast_stats.py`):
```python
import sys
from unittest.mock import MagicMock
sys.modules['freesasa'] = MagicMock()

import atom3d.datasets as da  # Will now import successfully without compiling freesasa
```

---

## 3. Verify the Installation
To make sure all encoders and dependencies load correctly, run:
```bash
python -c "from model import Encoder, PaiNNEncoder, DimeNetPlusPlusEncoder; print('All systems nominal!')"
```

---

## 4. Resuming the Project Context
Read [PROJECT_CONTEXT.md](file:///c:/Users/Dhanush/Desktop/project/PROJECT_CONTEXT.md) for details on the model architectures, dataset files, and QM9 evaluation metrics.
