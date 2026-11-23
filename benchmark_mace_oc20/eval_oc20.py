import os
import sys
import json
import time
import numpy as np
import ase.io
import torch
from mace.calculators import MACECalculator

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

OC20_MODEL_PATH = "/nobackup/proj/disk/materialsdesigndivision/personal/paul/training_mace_head4/mace_ft_tio2_cobalt.model"

PRIMARY_TEST_XYZ = os.path.join(PROJECT_ROOT, "dataset_preparation/output/test.xyz")
PRIMARY_NEB_TRAJ = os.path.join(PROJECT_ROOT, "main_project/nebs/neb_1/final.traj")

DIMER_TEST_XYZ = os.path.join(PROJECT_ROOT, "benchmark_dimer/output/test_dimer.xyz")
DIMER_NEB_DIR = os.path.join(PROJECT_ROOT, "benchmark_dimer/output/nebs")

def compute_metrics(dft_energies, pred_energies, dft_forces, pred_forces, natoms_list, elements_list):
    dft_e = np.array(dft_energies)
    pred_e = np.array(pred_energies)
    natoms = np.array(natoms_list)
    
    shift = np.mean(dft_e - pred_e)
    pred_e_aligned = pred_e + shift
    
    energy_errors_per_atom = (pred_e_aligned - dft_e) / natoms
    e_mae_per_atom = float(np.mean(np.abs(energy_errors_per_atom))) * 1000.0  # meV/atom
    e_rmse_per_atom = float(np.sqrt(np.mean(energy_errors_per_atom**2))) * 1000.0  # meV/atom
    
    ss_res = np.sum((dft_e - pred_e_aligned) ** 2)
    ss_tot = np.sum((dft_e - np.mean(dft_e)) ** 2)
    r2_energy = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
    
    dft_f = np.concatenate(dft_forces, axis=0)
    pred_f = np.concatenate(pred_forces, axis=0)
    
    force_diff = pred_f - dft_f
    force_mae = float(np.mean(np.abs(force_diff)))
    force_rmse = float(np.sqrt(np.mean(force_diff**2)))
    
    norm_dft = np.linalg.norm(dft_f, axis=1)
    norm_pred = np.linalg.norm(pred_f, axis=1)
    mask = (norm_dft > 1e-4) & (norm_pred > 1e-4)
    if np.any(mask):
        cos_sim = np.sum(dft_f[mask] * pred_f[mask], axis=1) / (norm_dft[mask] * norm_pred[mask])
        mean_cos_sim = float(np.mean(cos_sim))
    else:
        mean_cos_sim = 0.0
        
    all_elements = []
    for el_list in elements_list:
        all_elements.extend(el_list)
    all_elements = np.array(all_elements)
    
    element_force_mae = {}
    for sym in np.unique(all_elements):
        el_mask = (all_elements == sym)
        if np.any(el_mask):
            el_mae = float(np.mean(np.abs(force_diff[el_mask])))
            element_force_mae[str(sym)] = el_mae
            
    return {
        'energy_mae_mev_per_atom': e_mae_per_atom,
        'energy_rmse_mev_per_atom': e_rmse_per_atom,
        'energy_r2': r2_energy,
        'energy_shift_ev': float(shift),
        'force_mae_ev_per_ang': force_mae,
        'force_rmse_ev_per_ang': force_rmse,
        'force_cosine_similarity': mean_cos_sim,
        'element_force_mae_ev_per_ang': element_force_mae
    }

def evaluate_neb_pathway(calc, neb_path):
    if not os.path.exists(neb_path):
        return None
    images = ase.io.read(neb_path, index=':')
    dft_energies = [atoms.get_potential_energy() for atoms in images]
    pred_energies = []
    for atoms in images:
        atoms_copy = atoms.copy()
        atoms_copy.calc = calc
        pred_energies.append(atoms_copy.get_potential_energy())
        
    dft_e = np.array(dft_energies)
    pred_e = np.array(pred_energies)
    
    dft_rel = dft_e - dft_e[0]
    pred_rel = pred_e - pred_e[0]
    
    dft_barrier = float(np.max(dft_rel))
    pred_barrier = float(np.max(pred_rel))
    barrier_error = float(abs(pred_barrier - dft_barrier))
    
    dft_rxn_e = float(dft_rel[-1])
    pred_rxn_e = float(pred_rel[-1])
    rxn_e_error = float(abs(pred_rxn_e - dft_rxn_e))
    
    profile_mae = float(np.mean(np.abs(pred_rel - dft_rel)))
    
    return {
        'num_images': len(images),
        'dft_barrier_ev': dft_barrier,
        'pred_barrier_ev': pred_barrier,
        'barrier_error_ev': barrier_error,
        'dft_rxn_energy_ev': dft_rxn_e,
        'pred_rxn_energy_ev': pred_rxn_e,
        'rxn_energy_error_ev': rxn_e_error,
        'profile_mae_ev': profile_mae,
        'dft_rel_energies': dft_rel.tolist(),
        'pred_rel_energies': pred_rel.tolist()
    }

def evaluate_dataset(calc, xyz_path, tag):
    print(f"\nLoading {tag} dataset: {xyz_path}")
    frames = ase.io.read(xyz_path, index=':')
    total_frames = len(frames)
    print(f"Loaded {total_frames} frames.")
    
    dft_energies = [f.get_potential_energy() for f in frames]
    dft_forces = [f.get_forces() for f in frames]
    natoms_list = [len(f) for f in frames]
    elements_list = [f.get_chemical_symbols() for f in frames]
    categories = [f.info.get('category', 'general') for f in frames]
    
    final_npz = os.path.join(OUTPUT_DIR, f"eval_data_{tag}.npz")
    ckpt_npz = os.path.join(OUTPUT_DIR, f"checkpoint_{tag}.npz")
    
    start_idx = 0
    pred_energies = []
    pred_forces = []
    
    if os.path.exists(final_npz):
        print(f"Loading already computed final evaluation from: {final_npz}")
        loaded = np.load(final_npz)
        pred_energies = loaded['pred_energies'].tolist()
        p_forces_flat = loaded['pred_forces']
        f_offset = 0
        for k in range(total_frames):
            nat = natoms_list[k]
            pred_forces.append(p_forces_flat[f_offset:f_offset + nat])
            f_offset += nat
        start_idx = total_frames
    elif os.path.exists(ckpt_npz):
        try:
            ckpt = np.load(ckpt_npz)
            pred_energies = ckpt['pred_energies'].tolist()
            ckpt_f_flat = ckpt['pred_forces']
            start_idx = len(pred_energies)
            f_offset = 0
            for k in range(start_idx):
                nat = natoms_list[k]
                pred_forces.append(ckpt_f_flat[f_offset:f_offset + nat])
                f_offset += nat
            print(f"Resuming from checkpoint: {start_idx}/{total_frames} frames already computed.")
        except Exception as ex:
            print(f"Warning: could not read checkpoint ({ex}), starting fresh.")
            start_idx = 0
            pred_energies = []
            pred_forces = []
            
    t0 = time.time()
    for i in range(start_idx, total_frames):
        frame = frames[i]
        if (i + 1) % 250 == 0 or i == total_frames - 1:
            elapsed = time.time() - t0
            frames_done = i + 1 - start_idx
            fps = frames_done / elapsed if elapsed > 0 else 0
            eta_s = (total_frames - (i + 1)) / fps if fps > 0 else 0
            print(f"  [{i + 1}/{total_frames}] {fps:.1f} frames/s, ETA: {eta_s:.0f}s...")
            sys.stdout.flush()
            
        frame_eval = frame.copy()
        frame_eval.calc = calc
        pred_energies.append(frame_eval.get_potential_energy())
        pred_forces.append(frame_eval.get_forces())
        
        if (i + 1) % 250 == 0:
            np.savez(
                ckpt_npz,
                pred_energies=np.array(pred_energies),
                pred_forces=np.concatenate(pred_forces, axis=0)
            )
            
    t_total = time.time() - t0
    frames_done = total_frames - start_idx
    fps_avg = frames_done / t_total if t_total > 0 else 0
    print(f"Completed {tag} evaluation in {t_total:.1f} s ({fps_avg:.1f} frames/s)")
    
    np.savez_compressed(
        final_npz,
        dft_energies=np.array(dft_energies),
        pred_energies=np.array(pred_energies),
        dft_forces=np.concatenate(dft_forces, axis=0),
        pred_forces=np.concatenate(pred_forces, axis=0),
        categories=np.array(categories),
        elements=np.concatenate([np.array(e) for e in elements_list])
    )
    print(f"Saved evaluation arrays to: {final_npz}")
    if os.path.exists(ckpt_npz):
        try:
            os.remove(ckpt_npz)
        except OSError:
            pass
            
    # Metrics
    global_metrics = compute_metrics(
        dft_energies, pred_energies, dft_forces, pred_forces, natoms_list, elements_list
    )
    
    cat_metrics = {}
    for c in sorted(list(set(categories))):
        c_indices = [idx for idx, cat in enumerate(categories) if cat == c]
        if not c_indices:
            continue
        cat_metrics[c] = compute_metrics(
            [dft_energies[idx] for idx in c_indices],
            [pred_energies[idx] for idx in c_indices],
            [dft_forces[idx] for idx in c_indices],
            [pred_forces[idx] for idx in c_indices],
            [natoms_list[idx] for idx in c_indices],
            [elements_list[idx] for idx in c_indices]
        )
        
    return global_metrics, cat_metrics

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 75)
    print(f"Benchmarking Fine-Tuned OC20 MACE Model (Device: {device})")
    print(f"Model: {OC20_MODEL_PATH}")
    print("=" * 75)
    
    calc = MACECalculator(
        model_paths=OC20_MODEL_PATH,
        device=device,
        default_dtype="float64"
    )
    
    results = {
        'benchmark_info': {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'model_path': OC20_MODEL_PATH,
            'foundation_head': 'oc20_usemppbe',
            'device': device
        },
        'datasets': {}
    }
    
    # 1. Evaluate Primary Test Set (5,807 frames)
    if os.path.exists(PRIMARY_TEST_XYZ):
        gm_prim, cat_prim = evaluate_dataset(calc, PRIMARY_TEST_XYZ, "oc20_primary_test")
        print("\nEvaluating Primary NEB pathway...")
        neb_prim = evaluate_neb_pathway(calc, PRIMARY_NEB_TRAJ)
        results['datasets']['primary_test'] = {
            'xyz_path': PRIMARY_TEST_XYZ,
            'global_metrics': gm_prim,
            'category_metrics': cat_prim,
            'neb_pathway': neb_prim
        }
        
    # 2. Evaluate Dimer Test Set (10,660 frames)
    if os.path.exists(DIMER_TEST_XYZ):
        gm_dimer, cat_dimer = evaluate_dataset(calc, DIMER_TEST_XYZ, "oc20_dimer_test")
        print("\nEvaluating Dimer NEB pathways...")
        dimer_nebs = {}
        for n_name in ['neb_21', 'neb_21_final', 'neb_22', 'neb_22_final']:
            p = os.path.join(DIMER_NEB_DIR, f"{n_name}.traj")
            if os.path.exists(p):
                dimer_nebs[n_name] = evaluate_neb_pathway(calc, p)
        results['datasets']['dimer_test'] = {
            'xyz_path': DIMER_TEST_XYZ,
            'global_metrics': gm_dimer,
            'category_metrics': cat_dimer,
            'neb_pathways': dimer_nebs
        }
        
    # Save JSON results
    res_json_path = os.path.join(OUTPUT_DIR, "benchmark_results_oc20.json")
    with open(res_json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved comprehensive benchmark results JSON: {res_json_path}")
    
    # Print summary tables
    print("\n" + "=" * 85)
    print("SUMMARY RESULTS: FINE-TUNED OC20 MACE MODEL")
    print("=" * 85)
    for ds_name, ds_data in results['datasets'].items():
        gm = ds_data['global_metrics']
        print(f"Dataset: {ds_name.upper()}")
        print(f"  Energy MAE:  {gm['energy_mae_mev_per_atom']:.2f} meV/atom")
        print(f"  Energy RMSE: {gm['energy_rmse_mev_per_atom']:.2f} meV/atom")
        print(f"  Energy R2:   {gm['energy_r2']:.4f}")
        print(f"  Energy Shift: {gm['energy_shift_ev']:.2f} eV")
        print(f"  Force MAE:   {gm['force_mae_ev_per_ang']:.4f} eV/Å ({gm['force_mae_ev_per_ang']*1000:.1f} meV/Å)")
        print(f"  Force RMSE:  {gm['force_rmse_ev_per_ang']:.4f} eV/Å ({gm['force_rmse_ev_per_ang']*1000:.1f} meV/Å)")
        print(f"  Force Cos:   {gm['force_cosine_similarity']:.3f}")
        print("  Per-Element Force MAE (eV/Å):", {k: round(v, 4) for k, v in gm['element_force_mae_ev_per_ang'].items()})
        if 'neb_pathway' in ds_data and ds_data['neb_pathway']:
            neb = ds_data['neb_pathway']
            print(f"  Primary NEB: Barrier Error = {neb['barrier_error_ev']:.4f} eV, Rxn E Error = {neb['rxn_energy_error_ev']:.4f} eV, Profile MAE = {neb['profile_mae_ev']:.4f} eV")
        if 'neb_pathways' in ds_data:
            print("  Dimer NEB Pathways:")
            for nb_name, nb in ds_data['neb_pathways'].items():
                print(f"    - {nb_name:15s}: Barrier Error = {nb['barrier_error_ev']:.4f} eV, Rxn E Error = {nb['rxn_energy_error_ev']:.4f} eV, Profile MAE = {nb['profile_mae_ev']:.4f} eV")
        print("-" * 85)

if __name__ == '__main__':
    main()
