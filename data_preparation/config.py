import os

# Root directories to scan
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATA_SOURCES = [
    os.path.join(PROJECT_ROOT, 'main_project'),
    os.path.join(PROJECT_ROOT, 'writing')
]

OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Functional signature: strictly rev-vdW-DF2
CANONICAL_FUNCTIONAL = {
    'GGA': 'MK',
    'LUSE_VDW': True,
    'PARAM1': 0.123457,
    'PARAM2': 0.7114,
    'ZAB_VDW': -1.8867
}

# Chemical elements allowed in dataset
ALLOWED_ELEMENTS = {'Ti', 'O', 'Co', 'Br', 'C', 'H'}

# Quality thresholds
MAX_FORCE_OUTLIER = 25.0  # eV/Å, reject unphysical overlaps

# Extended dataset sampling parameters
RELAX_RMSD_THRESHOLD = 0.02       # Å minimum coordinate displacement to retain step
RELAX_ENERGY_DELTA_THRESHOLD = 0.01 # eV minimum energy delta to retain step
ALWAYS_KEEP_RELAX_ENDPOINTS = True

# Splits
SPLIT_RATIOS = {
    'train': 0.80,
    'val': 0.10,
    'test': 0.10
}
RANDOM_SEED = 42

# Isolated atom reference energies computed with rev-vdW-DF2 (spin-polarized, ENCUT=475 eV)
REFERENCE_ENERGIES_REV_VDW = {
    'H': -0.868106,
    'C': -0.365395,
    'O': -0.289323,
    'Ti': -0.910161,
    'Co': 0.545919,
    'Br': 1.498177
}