import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

PRIMARY_FT_OMAT_NPZ = os.path.join(PROJECT_ROOT, "benchmark_mace/output/eval_data_finetuned_mace.npz")
PRIMARY_BASE_OC20_NPZ = os.path.join(PROJECT_ROOT, "benchmark_mace/output/eval_data_oc20_usemppbe.npz")
DIMER_FT_OMAT_NPZ = os.path.join(PROJECT_ROOT, "benchmark_dimer/output/eval_data_finetuned_mace.npz")

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'lines.linewidth': 2.0,
    'lines.markersize': 7
})

def plot_primary_comparison(results):
    npz_oc20 = os.path.join(OUTPUT_DIR, "eval_data_oc20_primary_test.npz")
    if not os.path.exists(npz_oc20):
        print("eval_data_oc20_primary_test.npz not found.")
        return
        
    data_oc20 = np.load(npz_oc20)
    dft_e = data_oc20['dft_energies']
    pred_e_oc20 = data_oc20['pred_energies']
    dft_f = data_oc20['dft_forces'].flatten()
    pred_f_oc20 = data_oc20['pred_forces'].flatten()
    
    gm_oc20 = results['datasets']['primary_test']['global_metrics']
    shift_oc20 = gm_oc20['energy_shift_ev']
    pred_e_oc20_aligned = pred_e_oc20 + shift_oc20
    
    # Check if fine-tuned omat is available
    has_omat = os.path.exists(PRIMARY_FT_OMAT_NPZ)
    if has_omat:
        data_omat = np.load(PRIMARY_FT_OMAT_NPZ)
        pred_e_omat = data_omat['pred_energies']
        shift_omat = np.mean(dft_e - pred_e_omat)
        pred_e_omat_aligned = pred_e_omat + shift_omat
        pred_f_omat = data_omat['pred_forces'].flatten()
        
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    
    # 1. Energy Parity
    ax = axes[0]
    idx_e = np.random.choice(len(dft_e), min(3000, len(dft_e)), replace=False)
    if has_omat:
        ax.scatter(dft_e[idx_e], pred_e_omat_aligned[idx_e], color='#1f77b4', alpha=0.35, s=15, edgecolors='none', label='Fine-Tuned OMat Head')
    ax.scatter(dft_e[idx_e], pred_e_oc20_aligned[idx_e], color='#9467bd', alpha=0.45, s=18, edgecolors='none', label='Fine-Tuned OC20 Head')
    
#    lims = [min(np.min(dft_e), np.min(pred_e_oc20_aligned)) - 1.0, max(np.max(dft_e), np.max(pred_e_oc20_aligned)) + 1.0]
    lims = [-2850,-2700]
    ax.plot(lims, lims, 'k--', alpha=0.8, label='Ideal ($y=x$)')
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_title('Primary Test Set: Energy Parity (5,807 Frames)', fontweight='bold')
    ax.set_xlabel('DFT Energy (eV)'); ax.set_ylabel('MACE Energy [Aligned] (eV)')
    
    stats_text = (
        f"Fine-Tuned OC20:\n"
        f"  E-MAE:  {gm_oc20['energy_mae_mev_per_atom']:.2f} meV/at\n"
        f"  E-RMSE: {gm_oc20['energy_rmse_mev_per_atom']:.2f} meV/at\n"
        f"  $R^2$:    {gm_oc20['energy_r2']:.4f}\n"
        f"  Shift:  {shift_oc20:+.2f} eV"
    )
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.92, edgecolor='gray'), family='monospace', fontsize=9.5)
    ax.grid(True, linestyle=':', alpha=0.5); ax.legend(loc='lower right')
    
    # 2. Force Hexbin
    ax = axes[1]
    idx_f = np.random.choice(len(dft_f), min(25000, len(dft_f)), replace=False)
    lim_f = 3.5
    hb = ax.hexbin(dft_f[idx_f], pred_f_oc20[idx_f], gridsize=48, cmap='Purples', mincnt=1,
                   extent=[-lim_f, lim_f, -lim_f, lim_f], bins='log')
    cb = fig.colorbar(hb, ax=ax, shrink=0.85); cb.set_label('$\log_{10}(\mathrm{Count})$')
    ax.plot([-lim_f, lim_f], [-lim_f, lim_f], 'k--', alpha=0.8, label='Ideal ($y=x$)')
    ax.set_xlim([-lim_f, lim_f]); ax.set_ylim([-lim_f, lim_f])
    ax.set_title('Primary Test Set: Force Component Parity', fontweight='bold')
    ax.set_xlabel('DFT Force Component (eV/Å)'); ax.set_ylabel('MACE Force Component (eV/Å)')
    
    f_stats_text = (
        f"Fine-Tuned OC20:\n"
        f"  F-MAE:  {gm_oc20['force_mae_ev_per_ang']:.4f} eV/Å\n"
        f"  F-RMSE: {gm_oc20['force_rmse_ev_per_ang']:.4f} eV/Å\n"
        f"  Cosine: {gm_oc20['force_cosine_similarity']:.3f}"
    )
    ax.text(0.05, 0.95, f_stats_text, transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.92, edgecolor='gray'), family='monospace', fontsize=9.5)
    ax.grid(True, linestyle=':', alpha=0.5); ax.legend(loc='lower right')
    
    plt.tight_layout()
    p1 = os.path.join(OUTPUT_DIR, "oc20_primary_parity.png")
    plt.savefig(p1, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {p1}")

def plot_dimer_parity(results):
    npz_dimer = os.path.join(OUTPUT_DIR, "eval_data_oc20_dimer_test.npz")
    if not os.path.exists(npz_dimer):
        print("eval_data_oc20_dimer_test.npz not found.")
        return
        
    data = np.load(npz_dimer)
    dft_e = data['dft_energies']
    pred_e = data['pred_energies']
    dft_f = data['dft_forces'].flatten()
    pred_f = data['pred_forces'].flatten()
    
    gm = results['datasets']['dimer_test']['global_metrics']
    shift = gm['energy_shift_ev']
    pred_e_aligned = pred_e + shift
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    
    # 1. Energy
    ax = axes[0]
    idx_e = np.random.choice(len(dft_e), min(3000, len(dft_e)), replace=False)
    ax.scatter(dft_e[idx_e], pred_e_aligned[idx_e], color='#9467bd', alpha=0.45, s=18, edgecolors='none', label='Dimer Calculations')
#    lims = [min(np.min(dft_e), np.min(pred_e_aligned)) - 1.0, max(np.max(dft_e), np.max(pred_e_aligned)) + 1.0]
    lims = [-2850,-2700]
    ax.plot(lims, lims, 'k--', alpha=0.8, label='Ideal ($y=x$)')
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_title('Dimer Test Set: Energy Parity (10,660 Frames)', fontweight='bold')
    ax.set_xlabel('DFT Energy (eV)'); ax.set_ylabel('MACE Energy [Aligned] (eV)')
    
    stats_text = (
        f"Fine-Tuned OC20:\n"
        f"  E-MAE:  {gm['energy_mae_mev_per_atom']:.2f} meV/at\n"
        f"  E-RMSE: {gm['energy_rmse_mev_per_atom']:.2f} meV/at\n"
        f"  $R^2$:    {gm['energy_r2']:.4f}\n"
        f"  Shift:  {shift:+.2f} eV"
    )
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.92, edgecolor='gray'), family='monospace', fontsize=9.5)
    ax.grid(True, linestyle=':', alpha=0.5); ax.legend(loc='lower right')
    
    # 2. Force
    ax = axes[1]
    idx_f = np.random.choice(len(dft_f), min(25000, len(dft_f)), replace=False)
    lim_f = 3.5
    hb = ax.hexbin(dft_f[idx_f], pred_f[idx_f], gridsize=48, cmap='Purples', mincnt=1,
                   extent=[-lim_f, lim_f, -lim_f, lim_f], bins='log')
    cb = fig.colorbar(hb, ax=ax, shrink=0.85); cb.set_label('$\log_{10}(\mathrm{Count})$')
    ax.plot([-lim_f, lim_f], [-lim_f, lim_f], 'k--', alpha=0.8, label='Ideal ($y=x$)')
    ax.set_xlim([-lim_f, lim_f]); ax.set_ylim([-lim_f, lim_f])
    ax.set_title('Dimer Test Set: Force Component Parity', fontweight='bold')
    ax.set_xlabel('DFT Force Component (eV/Å)'); ax.set_ylabel('MACE Force Component (eV/Å)')
    
    f_stats_text = (
        f"Fine-Tuned OC20:\n"
        f"  F-MAE:  {gm['force_mae_ev_per_ang']:.4f} eV/Å\n"
        f"  F-RMSE: {gm['force_rmse_ev_per_ang']:.4f} eV/Å\n"
        f"  Cosine: {gm['force_cosine_similarity']:.3f}"
    )
    ax.text(0.05, 0.95, f_stats_text, transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.92, edgecolor='gray'), family='monospace', fontsize=9.5)
    ax.grid(True, linestyle=':', alpha=0.5); ax.legend(loc='lower right')
    
    plt.tight_layout()
    p2 = os.path.join(OUTPUT_DIR, "oc20_dimer_parity.png")
    plt.savefig(p2, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {p2}")

def plot_element_force_comparison(results):
    elements = ['Ti', 'O', 'Co', 'Br', 'C', 'H']
    oc20_gm = results['datasets']['primary_test']['global_metrics']['element_force_mae_ev_per_ang']
    
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    
    has_omat = os.path.exists(PRIMARY_FT_OMAT_NPZ)
    if has_omat:
        data_omat = np.load(PRIMARY_FT_OMAT_NPZ)
        f_diff = np.abs(data_omat['pred_forces'] - data_omat['dft_forces'])
        all_els = data_omat['elements']
        omat_el_mae = {}
        for el in elements:
            m = (all_els == el)
            omat_el_mae[el] = float(np.mean(f_diff[m])) if np.any(m) else 0.0
            
    x = np.arange(len(elements))
    bar_width = 0.35 if has_omat else 0.55
    
    if has_omat:
        bars1 = ax.bar(x - bar_width/2, [omat_el_mae.get(e, 0.0) for e in elements], bar_width,
                       label='Fine-Tuned (from OMat PBE)', color='#1f77b4', edgecolor='black', linewidth=0.8)
        bars2 = ax.bar(x + bar_width/2, [oc20_gm.get(e, 0.0) for e in elements], bar_width,
                       label='Fine-Tuned (from OC20 PBE)', color='#9467bd', edgecolor='black', linewidth=0.8)
        for bar in bars1:
            h = bar.get_height()
            if h > 0:
                ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 3),
                            textcoords="offset points", ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            h = bar.get_height()
            if h > 0:
                ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 3),
                            textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')
    else:
        bars = ax.bar(x, [oc20_gm.get(e, 0.0) for e in elements], bar_width,
                      label='Fine-Tuned (from OC20 PBE)', color='#9467bd', edgecolor='black', linewidth=0.8)
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 3),
                            textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
                            
    ax.set_title('Per-Element Force Mean Absolute Error (Primary Test Set)', fontweight='bold', pad=12)
    ax.set_xlabel('Chemical Element'); ax.set_ylabel('Force MAE (eV/Å)')
    ax.set_xticks(x); ax.set_xticklabels(elements, fontweight='bold', fontsize=12)
    ax.grid(True, axis='y', linestyle=':', alpha=0.6)
    ax.legend(loc='upper right', framealpha=0.9)
    
    ylim = ax.get_ylim()
    ax.set_ylim([0, ylim[1] * 1.15])
    
    plt.tight_layout()
    p3 = os.path.join(OUTPUT_DIR, "oc20_element_force_comparison.png")
    plt.savefig(p3, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {p3}")

def main():
    json_path = os.path.join(OUTPUT_DIR, "benchmark_results_oc20.json")
    if not os.path.exists(json_path):
        print(f"Results file not found: {json_path}")
        return
    with open(json_path, 'r') as f:
        results = json.load(f)
        
    print("Generating OC20 benchmark plots...")
    if 'primary_test' in results.get('datasets', {}):
        plot_primary_comparison(results)
        plot_element_force_comparison(results)
    if 'dimer_test' in results.get('datasets', {}):
        plot_dimer_parity(results)
    print("All OC20 plots generated successfully.")

if __name__ == '__main__':
    main()
