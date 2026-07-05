import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_DATA_DIR = os.path.join(BASE_DIR, "data")
PDB_DIR = os.path.join(RAW_DATA_DIR, "protein")
SDF_DIR = os.path.join(RAW_DATA_DIR, "molecule")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
LOG_DIR = os.path.join(BASE_DIR, "logs")

ALLOWED_ELEMENTS = ["C", "N", "O", "S", "P", "H", "F", "Cl", "Br", "I"]
MIN_HEAVY_ATOMS = 4
MAX_HEAVY_ATOMS = 1000
MAX_PEPTIDE_LENGTH = 120
PLDDT_CUTOFF = 70.0

DISTANCE_CUTOFF = 5.0
RBF_NUM_CENTERS = 16
RBF_MIN = 0.0
RBF_MAX = DISTANCE_CUTOFF
RBF_WIDTH = (RBF_MAX - RBF_MIN) / RBF_NUM_CENTERS
USE_VIRTUAL_NODE = True

NOISE_SCALE = 0.2

HIDDEN_DIM = 128
NUM_GNN_LAYERS = 4
EMBEDDING_DIM = 256
POOLING = "mean"
USE_JUMPING_KNOWLEDGE = False
SCHNET_NUM_FILTERS = 128
SCHNET_NUM_INTERACTIONS = 4

MASK_RATE = 0.15
USE_DESCRIPTOR_HEAD = False
DESCRIPTOR_LIST = ["logP", "TPSA", "MW"]
DESCRIPTOR_LOSS_WEIGHT = 0.1

LEARNING_RATE = 1e-3
BATCH_SIZE = 32
NUM_EPOCHS = 1000
WEIGHT_DECAY = 1e-5

try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    DEVICE = "cpu"

NUM_WORKERS = 4
VAL_SPLIT = 0.2
CHECKPOINT_EVERY = 5
RANDOM_SEED = 42
ENCODER_TYPE = "schnet"
MACE_MAX_ELL = 1
MACE_CORRELATION = 3
MACE_NUM_INTERACTIONS = 2
DEBUG_MODE = True
SUBSET_SIZE = 250

ENCODER_CHECKPOINT = os.path.join(BASE_DIR, "encoder_epoch_40.pt")
QM9_DIR = os.path.join(BASE_DIR, "data", "qm9")
QM9_TARGET_IDX = 4
QM9_TARGET_NAME = "HOMO-LUMO gap"
QM9_TARGET_UNIT = "eV"


os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PDB_DIR, exist_ok=True)
os.makedirs(SDF_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
