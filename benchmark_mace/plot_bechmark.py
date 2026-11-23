import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

# Set clean aesthetic styling
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.titlesize': 16,
    'lines.linewidth': 2.0,
    'lines.markersize': 7
})

HEAD_COLORS = {
    'omat_pbe': '#1f77b4',          # Blue
    'mp_pbe_refit_add': '#2ca02c',    # Green
    'matpes_r2scan': '#ff7f0e',      # Orange
    'oc20_usemppbe': '#9467bd',       # Purple
}

HEAD_LABELS = {
    'omat_pbe': 'OMat PBE',
    'mp_pbe_refit_add': 'MP PBE (Refit)',
    'matpes_r2scan': 'MatPES r2SCAN',
    'oc20_usemppbe': 'OC20 PBE',
}

def load_results():
    results_path = os.path.join(OUTPUT_DIR, 'benchmark_results.json')
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results file not found: {results_path}")
    with open(results_path, 'r') as f:
        return json.load(f)

def plot_energy_parity(results):
    heads = list(results['heads'].keys())
    n_heads = len(heads)

    fig, axes = plt.subplots(1, n_heads, figsize=(5.5 * n_heads, 5.0), squeeze=False)
    axes = axes.flatten()

    for idx, head in enumerate(heads):
        ax = axes[idx]
        npz_path = os.path.join(OUTPUT_DIR, f"eval_data_{head}.npz")

        if os.path.exists(npz_path):
            data = np.load(npz_path)
            dft_e = data['dft_energies']
            pred_e = data['pred_energies']
            shift = results['heads'][head]['global_metrics']['energy_shift_ev']
            pred_e_aligned = pred_e + shift

            # Subsample for plotting if large
            if len(dft_e) > 3000:
                sample_idx = np.random.choice(len(dft_e), 3000, replace=False)
                dft_plot = dft_e[sample_idx]
                pred_plot = pred_e_aligned[sample_idx]
            else:
                dft_plot = dft_e
                pred_plot = pred_e_aligned

            ax.scatter(dft_plot, pred_plot, alpha=0.35, s=15,
                       color=HEAD_COLORS.get(head, '#333333'), edgecolors='none')

            min_val = min(np.min(dft_plot), np.min(pred_plot))
            max_val = max(np.max(dft_plot), np.max(pred_plot))
            margin = 0.03 * (max_val - min_val)
            lims = [-2850, -2700]

            ax.plot(lims, lims, 'k--', alpha=0.7, label='Ideal ($y=x$)')
            ax.set_xlim(lims)
            ax.set_ylim(lims)

        m = results['heads'][head]['global_metrics']
        stats_text = (
            f"MAE:  {m['energy_mae_mev_per_atom']:.1f} meV/at\n"
            f"RMSE: {m['energy_rmse_mev_per_atom']:.1f} meV/at\n"
            f"$R^2$:   {m['energy_r2']:.4f}\n"
            f"Shift: {m['energy_shift_ev']:.1f} eV"
        )
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'),
                fontsize=10, family='monospace')

        ax.set_title(HEAD_LABELS.get(head, head), fontweight='bold')
        ax.set_xlabel('DFT Energy (eV)')
        if idx == 0:
            ax.set_ylabel('MACE Energy [Aligned] (eV)')
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.legend(loc='lower right', framealpha=0.8)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'energy_parity.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

def plot_force_parity(results):
    heads = list(results['heads'].keys())
    n_heads = len(heads)

    fig, axes = plt.subplots(1, n_heads, figsize=(5.5 * n_heads, 5.0), squeeze=False)
    axes = axes.flatten()

    for idx, head in enumerate(heads):
        ax = axes[idx]
        npz_path = os.path.join(OUTPUT_DIR, f"eval_data_{head}.npz")

        if os.path.exists(npz_path):
            data = np.load(npz_path)
            dft_f = data['dft_forces'].flatten()
            pred_f = data['pred_forces'].flatten()

            # Subsample 15,000 points for hexbin
            if len(dft_f) > 15000:
                sample_idx = np.random.choice(len(dft_f), 15000, replace=False)
                dft_plot = dft_f[sample_idx]
                pred_plot = pred_f[sample_idx]
            else:
                dft_plot = dft_f
                pred_plot = pred_f

            lim = 3.5
            hb = ax.hexbin(dft_plot, pred_plot, gridsize=45, cmap='Blues',
                           mincnt=1, extent=[-lim, lim, -lim, lim], bins='log')
            cb = fig.colorbar(hb, ax=ax, shrink=0.8)
            cb.set_label('log10(Count)')

            ax.plot([-lim, lim], [-lim, lim], 'r--', alpha=0.8, label='Ideal ($y=x$)')
            ax.set_xlim([-lim, lim])
            ax.set_ylim([-lim, lim])

        m = results['heads'][head]['global_metrics']
        stats_text = (
            f"F-MAE:  {m['force_mae_ev_per_ang']:.3f} eV/Å\n"
            f"F-RMSE: {m['force_rmse_ev_per_ang']:.3f} eV/Å\n"
            f"Cosine: {m['force_cosine_similarity']:.3f}"
        )
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'),
                fontsize=10, family='monospace')

        ax.set_title(HEAD_LABELS.get(head, head), fontweight='bold')
        ax.set_xlabel('DFT Force Component (eV/Å)')
        if idx == 0:
            ax.set_ylabel('MACE Force Component (eV/Å)')
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.legend(loc='lower right', framealpha=0.8)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'force_parity.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

def plot_reaction_coordinate(results):
    # Exclude fine-tuned model results as requested; compare foundation model heads vs DFT ground truth
    heads = [h for h in results['heads'].keys() if h != 'finetuned_mace' and 'finetun' not in h.lower()]

    plt.figure(figsize=(9, 6))

    # Check if first head has NEB pathway
    first_head = heads[0]
    neb_first = results['heads'][first_head].get('neb_pathway')
    if not neb_first:
        print("No NEB pathway data available for plotting.")
        return

    dft_rel = np.array(neb_first['dft_rel_energies'])
    rxn_coord = np.arange(len(dft_rel))

    # Ground truth DFT curve
    plt.plot(rxn_coord, dft_rel, 'k-o', linewidth=2.5, markersize=8, label='DFT Ground Truth (rev-vdW-DF2)', zorder=5)

    markers = ['s', '^', 'D', 'v', 'p']
    for idx, head in enumerate(heads):
        neb_data = results['heads'][head].get('neb_pathway')
        if neb_data:
            pred_rel = np.array(neb_data['pred_rel_energies'])
            color = HEAD_COLORS.get(head, f"C{idx}")
            label = f"{HEAD_LABELS.get(head, head)} (Err: $\\Delta E^\\ddagger$={neb_data['barrier_error_ev']:.2f} eV)"
            plt.plot(rxn_coord, pred_rel, linestyle='--', marker=markers[idx % len(markers)],
                     color=color, linewidth=2, markersize=7, label=label, zorder=4)

    # Annotate TS (Image 3)
    ts_idx = int(np.argmax(dft_rel))
    dft_barrier = dft_rel[ts_idx]
    plt.axvline(ts_idx, color='gray', linestyle=':', alpha=0.7)
    plt.text(ts_idx + 0.1, dft_barrier * 0.95, f"Transition State\nDFT: {dft_barrier:+.3f} eV",
             fontweight='bold', fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffffcc', alpha=0.8))

    # Annotate Product (Final image)
    prod_idx = len(dft_rel) - 1
    dft_rxn_e = dft_rel[prod_idx]
    plt.text(prod_idx - 0.8, dft_rxn_e + 0.15, f"Product\nDFT: {dft_rxn_e:+.3f} eV",
             fontweight='bold', fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='#e6f2ff', alpha=0.8))

    plt.title('NEB Reaction Pathway: C-Br Cleavage on Co/TiO2(110)', fontweight='bold', pad=15)
    plt.xlabel('Reaction Coordinate (NEB Image Index)')
    plt.ylabel('Relative Energy $\\Delta E$ (eV)')
    plt.xticks(rxn_coord, [f"Image {i:02d}" for i in rxn_coord], rotation=25)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower left', framealpha=0.9)
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, 'reaction_coordinate_profile.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

def plot_element_force_errors(results):
    heads = list(results['heads'].keys())

    # Collect all elements
    all_elements = set()
    for h in heads:
        el_dict = results['heads'][h]['global_metrics']['element_force_mae_ev_per_ang']
        all_elements.update(el_dict.keys())

    # Sort elements logically
    desired_order = ['Ti', 'O', 'Co', 'Br', 'C', 'H']
    elements = [e for e in desired_order if e in all_elements]
    # Add any extra elements
    for e in sorted(list(all_elements)):
        if e not in elements:
            elements.append(e)

    n_elements = len(elements)
    n_heads = len(heads)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bar_width = 0.8 / n_heads
    x = np.arange(n_elements)

    for h_idx, head in enumerate(heads):
        el_dict = results['heads'][head]['global_metrics']['element_force_mae_ev_per_ang']
        maes = [el_dict.get(e, 0.0) for e in elements]
        offset = (h_idx - (n_heads - 1) / 2) * bar_width
        bars = ax.bar(x + offset, maes, bar_width, label=HEAD_LABELS.get(head, head),
                      color=HEAD_COLORS.get(head, f"C{h_idx}"), edgecolor='black', linewidth=0.8)

        # Add labels on top of bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f"{height:.2f}",
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8, rotation=45)

    ax.set_title('Per-Element Force Mean Absolute Error across Heads', fontweight='bold', pad=15)
    ax.set_xlabel('Chemical Element')
    ax.set_ylabel('Force MAE (eV/Å)')
    ax.set_xticks(x)
    ax.set_xticklabels(elements, fontweight='bold', fontsize=12)
    ax.grid(True, axis='y', linestyle=':', alpha=0.6)
    ax.legend(loc='upper right', framealpha=0.9)

    # Leave room for bar top labels
    ylim = ax.get_ylim()
    ax.set_ylim([0, ylim[1] * 1.15])

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'element_force_errors.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

def plot_energy_error_distributions(results):
    heads = list(results['heads'].keys())

    fig, (ax_hist, ax_box) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={'width_ratios': [2, 1]})

    error_data = []
    labels = []
    colors = []

    for head in heads:
        npz_path = os.path.join(OUTPUT_DIR, f"eval_data_{head}.npz")
        if not os.path.exists(npz_path):
            continue
        data = np.load(npz_path)
        dft_e = data['dft_energies']
        pred_e = data['pred_energies']
        elements = data['elements']
        avg_natoms = len(elements) / len(dft_e)

        shift = results['heads'][head]['global_metrics']['energy_shift_ev']
        errors_mev = ((pred_e + shift) - dft_e) / avg_natoms * 1000.0

        # Clip extreme outliers for clean distribution visualization
        errors_clipped = errors_mev[np.abs(errors_mev) < 100.0]

        c = HEAD_COLORS.get(head, '#333333')
        label = f"{HEAD_LABELS.get(head, head)} (MAE: {results['heads'][head]['global_metrics']['energy_mae_mev_per_atom']:.1f})"

        ax_hist.hist(errors_clipped, bins=80, density=True, histtype='step',
                     linewidth=2.2, color=c, label=label, alpha=0.9)

        error_data.append(errors_mev)
        labels.append(HEAD_LABELS.get(head, head))
        colors.append(c)

    ax_hist.axvline(0, color='black', linestyle='--', alpha=0.6)
    ax_hist.set_title('Energy Error Distribution ($E_{pred} - E_{DFT}$)', fontweight='bold')
    ax_hist.set_xlabel('Energy Error (meV/atom)')
    ax_hist.set_ylabel('Probability Density')
    ax_hist.set_xlim([-50, 50])
    ax_hist.grid(True, linestyle=':', alpha=0.5)
    ax_hist.legend(loc='upper right', framealpha=0.9)

    # Box plot
    if error_data:
        bp = ax_box.boxplot(error_data, labels=labels, patch_artist=True, showfliers=False,
                            medianprops=dict(color='black', linewidth=1.5))
        for patch, c in zip(bp['boxes'], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.6)
        ax_box.axhline(0, color='black', linestyle='--', alpha=0.6)
        ax_box.set_title('Error Spread (IQR)', fontweight='bold')
        ax_box.set_ylabel('Energy Error (meV/atom)')
        ax_box.set_ylim([-40, 40])
        ax_box.grid(True, linestyle=':', alpha=0.5)
        ax_box.tick_params(axis='x', rotation=25)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'energy_error_distribution.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

def plot_category_energy_errors(results):
    heads = list(results['heads'].keys())

    # Target categories with substantial representation
    target_cats = [
        'neb_pathway',
        'reaction_intermediates',
        'adsorption',
        'vibrational_analysis',
        'cobalt_agglomeration'
    ]

    cat_names = {
        'neb_pathway': 'NEB Pathways',
        'reaction_intermediates': 'Intermediates',
        'adsorption': 'Adsorption States',
        'vibrational_analysis': 'Vibrations',
        'cobalt_agglomeration': 'Co Agglomeration'
    }

    n_cats = len(target_cats)
    n_heads = len(heads)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    bar_width = 0.8 / n_heads
    x = np.arange(n_cats)

    for h_idx, head in enumerate(heads):
        npz_path = os.path.join(OUTPUT_DIR, f"eval_data_{head}.npz")
        if not os.path.exists(npz_path):
            continue
        data = np.load(npz_path)
        dft_e = data['dft_energies']
        pred_e = data['pred_energies']
        cats = data['categories']
        elements = data['elements']
        avg_natoms = len(elements) / len(dft_e)

        maes = []
        for cat in target_cats:
            mask = (cats == cat)
            if np.any(mask):
                c_dft = dft_e[mask]
                c_pred = pred_e[mask]
                c_shift = np.mean(c_dft - c_pred)
                c_err = np.abs((c_pred + c_shift) - c_dft)
                mae_at = float(np.mean(c_err) / avg_natoms * 1000.0)
                maes.append(mae_at)
            else:
                maes.append(0.0)

        offset = (h_idx - (n_heads - 1) / 2) * bar_width
        bars = ax.bar(x + offset, maes, bar_width, label=HEAD_LABELS.get(head, head),
                      color=HEAD_COLORS.get(head, f"C{h_idx}"), edgecolor='black', linewidth=0.8)

        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f"{height:.1f}",
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9, rotation=35)

    ax.set_title('Category-Stratified Energy MAE across Heads', fontweight='bold', pad=15)
    ax.set_xlabel('Chemical Configuration Category')
    ax.set_ylabel('Energy MAE (meV/atom)')
    ax.set_xticks(x)
    ax.set_xticklabels([cat_names.get(c, c) for c in target_cats], fontweight='bold', fontsize=11)
    ax.grid(True, axis='y', linestyle=':', alpha=0.6)
    ax.legend(loc='upper right', framealpha=0.9)

    ylim = ax.get_ylim()
    ax.set_ylim([0, ylim[1] * 1.18])

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'category_energy_errors.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

def main():
    print("Generating benchmark publication plots...")
    results = load_results()
    plot_energy_parity(results)
    plot_energy_error_distributions(results)
    plot_category_energy_errors(results)
    plot_force_parity(results)
    plot_reaction_coordinate(results)
    plot_element_force_errors(results)
    print("All benchmark plots generated successfully.")

if __name__ == '__main__':
    main()