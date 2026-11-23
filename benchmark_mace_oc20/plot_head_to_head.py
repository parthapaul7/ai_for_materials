import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

# Load benchmark results
with open(os.path.join(OUTPUT_DIR, 'benchmark_results_oc20.json')) as f:
    oc20_res = json.load(f)

with open(os.path.join(PROJECT_ROOT, 'benchmark_mace/output/benchmark_results.json')) as f:
    mace_res = json.load(f)

with open(os.path.join(PROJECT_ROOT, 'benchmark_dimer/output/benchmark_results.json')) as f:
    dimer_res = json.load(f)

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.titlesize': 15,
    'lines.linewidth': 2.0
})

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Primary Test Set Metrics Comparison
ax = axes[0, 0]
models = ['Base OC20\n(usemppbe)', 'Fine-Tuned\nOMat Head', 'Fine-Tuned\nOC20 Head']
e_maes = [275.01, 1.82, oc20_res['datasets']['primary_test']['global_metrics']['energy_mae_mev_per_atom']]
f_maes = [80.4, 8.3, oc20_res['datasets']['primary_test']['global_metrics']['force_mae_ev_per_ang'] * 1000]

x = np.arange(len(models))
width = 0.35

ax2 = ax.twinx()
b1 = ax.bar(x - width/2, e_maes, width, label='Energy MAE (meV/at)', color='#4c72b0', edgecolor='black')
b2 = ax2.bar(x + width/2, f_maes, width, label='Force MAE (meV/Å)', color='#dd8452', edgecolor='black')

ax.set_yscale('log')
ax.set_ylabel('Energy MAE (meV/atom) [log scale]', color='#4c72b0', fontweight='bold')
ax2.set_ylabel('Force MAE (meV/Å)', color='#dd8452', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models, fontweight='bold')
ax.set_title('(a) Primary Test Set Accuracy (5,807 Frames)', fontweight='bold')

for bar in b1:
    h = bar.get_height()
    ax.annotate(f'{h:.2f}' if h < 10 else f'{h:.0f}', xy=(bar.get_x() + bar.get_width()/2, h),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
for bar in b2:
    h = bar.get_height()
    ax2.annotate(f'{h:.1f}', xy=(bar.get_x() + bar.get_width()/2, h),
                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

# Combine legends
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
ax.grid(True, linestyle=':', alpha=0.5)

# 2. Dimer Test Set Metrics Comparison
ax = axes[0, 1]
models_dimer = ['Base OMat\n(omat_pbe)', 'Fine-Tuned\nOMat Head', 'Fine-Tuned\nOC20 Head']
e_maes_d = [5.01, 1.37, oc20_res['datasets']['dimer_test']['global_metrics']['energy_mae_mev_per_atom']]
f_maes_d = [97.5, 12.6, oc20_res['datasets']['dimer_test']['global_metrics']['force_mae_ev_per_ang'] * 1000]

x = np.arange(len(models_dimer))
ax2 = ax.twinx()
b1 = ax.bar(x - width/2, e_maes_d, width, label='Energy MAE (meV/at)', color='#55a868', edgecolor='black')
b2 = ax2.bar(x + width/2, f_maes_d, width, label='Force MAE (meV/Å)', color='#c44e52', edgecolor='black')

ax.set_ylabel('Energy MAE (meV/atom)', color='#55a868', fontweight='bold')
ax2.set_ylabel('Force MAE (meV/Å)', color='#c44e52', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models_dimer, fontweight='bold')
ax.set_title('(b) Dimer Test Set Accuracy (10,660 Frames)', fontweight='bold')

for bar in b1:
    h = bar.get_height()
    ax.annotate(f'{h:.2f}', xy=(bar.get_x() + bar.get_width()/2, h),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
for bar in b2:
    h = bar.get_height()
    ax2.annotate(f'{h:.1f}', xy=(bar.get_x() + bar.get_width()/2, h),
                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
ax.grid(True, linestyle=':', alpha=0.5)

# 3. Per-Element Force MAE: OMat vs OC20 Fine-Tuned
ax = axes[1, 0]
elements = ['Ti', 'O', 'Co', 'Br', 'C', 'H']
# Primary test per-element
npz_omat = os.path.join(PROJECT_ROOT, 'benchmark_mace/output/eval_data_finetuned_mace.npz')
data_omat = np.load(npz_omat)
f_diff_omat = np.abs(data_omat['pred_forces'] - data_omat['dft_forces'])
omat_els = data_omat['elements']
omat_f_el = [float(np.mean(f_diff_omat[omat_els == el])) * 1000 for el in elements]

oc20_f_el = [oc20_res['datasets']['primary_test']['global_metrics']['element_force_mae_ev_per_ang'][el] * 1000 for el in elements]

x = np.arange(len(elements))
b1 = ax.bar(x - width/2, omat_f_el, width, label='Fine-Tuned OMat', color='#1f77b4', edgecolor='black')
b2 = ax.bar(x + width/2, oc20_f_el, width, label='Fine-Tuned OC20', color='#9467bd', edgecolor='black')

ax.set_ylabel('Force MAE (meV/Å)', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(elements, fontweight='bold', fontsize=12)
ax.set_title('(c) Per-Element Force MAE (Primary Test Set)', fontweight='bold')
ax.legend(loc='upper right')
ax.grid(True, axis='y', linestyle=':', alpha=0.5)

for bar in b1:
    h = bar.get_height()
    ax.annotate(f'{h:.1f}', xy=(bar.get_x() + bar.get_width()/2, h),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
for bar in b2:
    h = bar.get_height()
    ax.annotate(f'{h:.1f}', xy=(bar.get_x() + bar.get_width()/2, h),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')

# 4. Reaction Pathway Barrier Errors (Primary NEB 1 & Dimer NEBs)
ax = axes[1, 1]
pathways = ['NEB 1\n(Cobalt)', 'NEB 21\n(C-C Init)', 'NEB 21 Final\n(Coupling)', 'NEB 22\n(Direct)', 'NEB 22 Final\n(Coupling)']
omat_barr_err = [0.0244, 0.0244, 0.0005, 0.0244, 0.0161]
oc20_barr_err = [
    oc20_res['datasets']['primary_test']['neb_pathway']['barrier_error_ev'],
    oc20_res['datasets']['dimer_test']['neb_pathways']['neb_21']['barrier_error_ev'],
    oc20_res['datasets']['dimer_test']['neb_pathways']['neb_21_final']['barrier_error_ev'],
    oc20_res['datasets']['dimer_test']['neb_pathways']['neb_22']['barrier_error_ev'],
    oc20_res['datasets']['dimer_test']['neb_pathways']['neb_22_final']['barrier_error_ev']
]

x = np.arange(len(pathways))
b1 = ax.bar(x - width/2, [b * 1000 for b in omat_barr_err], width, label='Fine-Tuned OMat Head', color='#1f77b4', edgecolor='black')
b2 = ax.bar(x + width/2, [b * 1000 for b in oc20_barr_err], width, label='Fine-Tuned OC20 Head', color='#9467bd', edgecolor='black')

ax.set_ylabel('Barrier Absolute Error (meV)', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(pathways, fontweight='bold', fontsize=9.5)
ax.set_title('(d) NEB Reaction Pathway Activation Barrier Error', fontweight='bold')
ax.legend(loc='upper right')
ax.grid(True, axis='y', linestyle=':', alpha=0.5)

for bar in b1:
    h = bar.get_height()
    ax.annotate(f'{h:.1f}', xy=(bar.get_x() + bar.get_width()/2, h),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
for bar in b2:
    h = bar.get_height()
    ax.annotate(f'{h:.1f}', xy=(bar.get_x() + bar.get_width()/2, h),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.suptitle('Comprehensive Model Benchmark: Fine-Tuned OC20 vs Fine-Tuned OMat vs Base Models', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
out_p = os.path.join(OUTPUT_DIR, 'head_to_head_comparison.png')
plt.savefig(out_p, dpi=300, bbox_inches='tight')
plt.close()
print(f'Successfully saved: {out_p}')
