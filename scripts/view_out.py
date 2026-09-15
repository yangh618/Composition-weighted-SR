import matplotlib.pyplot as plt
import re
import matplotlib as mpl

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman', 'Times', 'Nimbus Roman', 'DejaVu Serif']
mpl.rcParams['mathtext.fontset'] = 'stix'   # makes math match Times style
mpl.rcParams['axes.unicode_minus'] = False


# APS Journal Settings (NO FONT WARNINGS)
plt.rcParams.update({
    'font.size': 20,
    'axes.linewidth': 1.5,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'figure.dpi': 120
})

LOG_FILES = [
    "experiments/expt_gap/benchmark_score/fold_01/OUT",
    "experiments/expt_is_metal/benchmark_score/fold_01/OUT",
    "experiments/glass/benchmark_score/fold_01/OUT"
]

labels = ['Expt Gap', 'Expt Is Metal', 'Glass']

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

for i, (ax, log_file, label) in enumerate(zip(axes, LOG_FILES, labels)):
    with open(log_file, "r") as f:
        log = f.read()

    matches = re.findall(r"Total Nodes: (\d+).*?MAE \(↓\): ([\d.e-]+).*?Best Expression:\s*(.*?)\s*Reward", log, re.DOTALL)

    nodes, mae, exprs = [], [], []
    for n, m, e in matches:
        nodes.append(int(n))
        mae.append(float(m))
        exprs.append(e.strip())

    turn_nodes, turn_mae, turn_exprs = [], [], []
    prev_mae = None
    for n, m, e in zip(nodes, mae, exprs):
        if m != prev_mae:
            turn_nodes.append(n)
            turn_mae.append(m)
            turn_exprs.append(e)
            prev_mae = m

    ax.plot(nodes, mae, c='k', lw=1.5)
    ax.scatter(turn_nodes, turn_mae, c='k', s=25, zorder=5)

    # # Sparse annotations (every 5 + last point)
    # step = 5
    # for j, (x, y, text) in enumerate(zip(turn_nodes, turn_mae, turn_exprs)):
    #     if j % step == 0 or j == len(turn_nodes)-1:
    #         ax.annotate(text, (x, y), xytext=(4,4), textcoords='offset points', fontsize=7)

    ax.set_xlabel('Total Nodes')
    ax.set_ylabel('MAE')
    ax.set_xscale('log')
    ax.text(0.05, 0.05, f'({chr(97+i)})', transform=ax.transAxes, fontsize=20, va='bottom')

plt.tight_layout()
plt.savefig('mae_nodes_aps.eps', dpi=120, bbox_inches='tight')
plt.show()