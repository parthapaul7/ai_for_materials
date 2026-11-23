import os
import re
import numpy as np
import ase.io
from ase.calculators.singlepoint import SinglePointCalculator
from config import ALLOWED_ELEMENTS

def find_incar_settings(directory):
    """
    Search directory and up to 3 parent directories for INCAR.
    Returns dictionary of INCAR parameters.
    """
    cur = os.path.abspath(directory)
    for _ in range(4):
        incar_path = os.path.join(cur, 'INCAR')
        if os.path.isfile(incar_path):
            settings = {
                'gga': None,
                'luse_vdw': False,
                'zab_vdw': None,
                'param1': None,
                'param2': None,
                'incar_path': incar_path
            }
            try:
                with open(incar_path, 'r', errors='ignore') as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                        if '=' in line:
                            key, val = line.split('=', 1)
                            key = key.strip().upper()
                            val = val.split('#')[0].strip()
                            if key == 'GGA':
                                settings['gga'] = val.upper()
                            elif key == 'LUSE_VDW':
                                settings['luse_vdw'] = '.TRUE.' in val.upper()
                            elif key == 'ZAB_VDW':
                                settings['zab_vdw'] = float(val)
                            elif key == 'PARAM1':
                                settings['param1'] = float(val)
                            elif key == 'PARAM2':
                                settings['param2'] = float(val)
                return settings
            except Exception:
                pass
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return None

def is_rev_vdw_df2(settings):
    """Check if settings correspond to rev-vdW-DF2 (GGA=MK, LUSE_VDW=True)."""
    if not settings:
        return False
    gga = settings.get('gga')
    vdw = settings.get('luse_vdw', False)
    if gga == 'MK' and vdw:
        return True
    return False

def classify_category(rel_path):
    """Categorize directory into chemical system/pathway stage."""
    low = rel_path.lower()
    if 'neb' in low or re.search(r'/\d{2}$', rel_path):
        return 'neb_pathway'
    elif any(k in low for k in ['1path', '2path', '3step', 'br_removal']):
        return 'reaction_intermediates'
    elif any(k in low for k in ['adsorption', 'co_sites', 'adsorp']):
        return 'adsorption'
    elif 'agglomer' in low:
        return 'cobalt_agglomeration'
    elif 'fragment' in low:
        return 'molecular_fragments'
    elif 'surface' in low or 'slab' in low:
        return 'surface_slab'
    elif 'vibrational' in low or 'zpe' in low:
        return 'vibrational_analysis'
    else:
        return 'general_relaxation'

def parse_outcar_frames(outcar_path, base_root):
    """
    Parse all valid ionic frames from an OUTCAR.
    Extracts raw unconstrained forces (bypassing selective dynamics zeroing),
    energies, stresses, and metadata.
    """
    rel_dir = os.path.relpath(os.path.dirname(outcar_path), base_root)
    calc_dir = os.path.dirname(outcar_path)
    incar_settings = find_incar_settings(calc_dir)

    if not is_rev_vdw_df2(incar_settings):
        return []

    category = classify_category(rel_dir)

    try:
        raw_frames = ase.io.read(outcar_path, index=':')
    except Exception:
        return []

    if not isinstance(raw_frames, list):
        raw_frames = [raw_frames]

    valid_frames = []
    total_frames = len(raw_frames)

    for idx, atoms in enumerate(raw_frames):
        # Verify chemical species
        symbols = set(atoms.get_chemical_symbols())
        if not symbols.issubset(ALLOWED_ELEMENTS):
            continue

        # Extract raw unconstrained forces
        if atoms.calc and 'forces' in atoms.calc.results:
            raw_forces = np.array(atoms.calc.results['forces'], dtype=float)
        else:
            try:
                atoms.set_constraint(None)
                raw_forces = atoms.get_forces()
            except Exception:
                continue

        if raw_forces is None or len(raw_forces) != len(atoms):
            continue

        # Extract energy
        try:
            energy = atoms.get_potential_energy()
        except Exception:
            continue

        # Strip selective dynamics constraints
        atoms.set_constraint(None)

        # Stress tensor (if present)
        stress = None
        if atoms.calc and 'stress' in atoms.calc.results:
            try:
                stress = np.array(atoms.calc.results['stress'], dtype=float)
            except Exception:
                stress = None

        # Attach SinglePointCalculator with raw physical quantities
        calc_kwargs = {'energy': float(energy), 'forces': raw_forces}
        if stress is not None:
            calc_kwargs['stress'] = stress
        atoms.calc = SinglePointCalculator(atoms, **calc_kwargs)

        # Set metadata in info
        atoms.info['source_dir'] = rel_dir
        atoms.info['frame_index'] = idx
        atoms.info['total_frames_in_run'] = total_frames
        atoms.info['category'] = category
        atoms.info['functional'] = 'rev-vdW-DF2'

        valid_frames.append(atoms)

    return valid_frames