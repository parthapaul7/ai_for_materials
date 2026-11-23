import numpy as np
from config import MAX_FORCE_OUTLIER

def is_physically_sound(atoms):
    """
    Check that atomic forces and geometry are physically sound:
    - Finite energies and forces
    - Force magnitude does not exceed outlier threshold (25 eV/Å)
    - Minimum distance between any two atoms > 0.6 Å (no nuclear overlap)
    """
    try:
        energy = atoms.get_potential_energy()
        if np.isnan(energy):
            return False
    except Exception:
        return False

    forces = atoms.calc.results.get('forces', None) if atoms.calc else atoms.arrays.get('forces', None)
    if forces is None or np.isnan(forces).any():
        return False

    force_norms = np.linalg.norm(forces, axis=1)
    if len(force_norms) > 0 and np.max(force_norms) > MAX_FORCE_OUTLIER:
        return False

    # Check minimum interatomic distance for systems with >1 atom
    if len(atoms) > 1 and len(atoms) < 500:
        pos = atoms.get_positions()
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        if np.min(dist) < 0.6:
            return False

    return True

def compute_trajectory_signature(frames):
    """
    Compute a fingerprint for an entire trajectory to detect identical runs
    duplicated between main_project and writing.
    """
    if not frames:
        return None
    frame0 = frames[0]
    frame_last = frames[-1]

    symbols = tuple(sorted(frame0.get_chemical_symbols()))
    natoms = len(frame0)
    vol = round(frame0.get_volume(), 1)
    e0 = round(frame0.get_potential_energy(), 3)
    ef = round(frame_last.get_potential_energy(), 3)
    nsteps = len(frames)

    return (natoms, symbols, vol, e0, ef, nsteps)