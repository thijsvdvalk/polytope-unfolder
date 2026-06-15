from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import json
import numpy as np

"""
Table with just average over the 19,000 polytopes

Plot with on the x-axis facet count, and y-axis success rate can be 
Each facet_dir is a bin, for each bin calculate avg per strat, plot that
"""


DATASET_DIR = Path("..") / "polytopes_dataset" / "normal_distribution"
facet_dirs = [f"{i}-{i+4}_facets" for i in range(5, 100, 5)]

heur = {
    "PriorityTraversalConfig: a=-1, b=0": np.zeros(len(facet_dirs)),
    "PriorityTraversalConfig: a=1, b=0": np.zeros(len(facet_dirs)),
    "PriorityTraversalConfig: a=0, b=-1": np.zeros(len(facet_dirs)),
    "PriorityTraversalConfig: a=0, b=1": np.zeros(len(facet_dirs)),
    "PriorityTraversalConfig: a=-0.5, b=-0.5": np.zeros(len(facet_dirs)),
    "PriorityTraversalConfig: a=0.5, b=0.5": np.zeros(len(facet_dirs)),
    "PriorityTraversalConfig: a=0.5, b=-0.5": np.zeros(len(facet_dirs)),
    "PriorityTraversalConfig: a=-0.5, b=0.5": np.zeros(len(facet_dirs)),
    "MSTConfig: a=-1, b=0, c=0": np.zeros(len(facet_dirs)),
    "MSTConfig: a=1, b=0, c=0": np.zeros(len(facet_dirs)),
    "MSTConfig: a=0, b=-1, c=0": np.zeros(len(facet_dirs)),
    "MSTConfig: a=0, b=1, c=0": np.zeros(len(facet_dirs)),
    "MSTConfig: a=0, b=0, c=-1": np.zeros(len(facet_dirs)),
    "MSTConfig: a=0, b=0, c=1": np.zeros(len(facet_dirs)),
    "MSTConfig: a=-0.5, b=-0.5, c=0": np.zeros(len(facet_dirs)),
    "MSTConfig: a=0.5, b=0.5, c=0": np.zeros(len(facet_dirs)),
    "MSTConfig: a=-0.5, b=0.5, c=0": np.zeros(len(facet_dirs)),
    "MSTConfig: a=0.5, b=-0.5, c=0": np.zeros(len(facet_dirs)),
    "MSTConfig: a=-0.5, b=0, c=-0.5": np.zeros(len(facet_dirs)),
    "MSTConfig: a=0.5, b=0, c=0.5": np.zeros(len(facet_dirs)),
    "MSTConfig: a=-0.5, b=0, c=0.5": np.zeros(len(facet_dirs)),
    "MSTConfig: a=0.5, b=0, c=-0.5": np.zeros(len(facet_dirs)),
    "MSTConfig: a=0, b=-0.5, c=-0.5": np.zeros(len(facet_dirs)),
    "MSTConfig: a=0, b=0.5, c=0.5": np.zeros(len(facet_dirs)),
    "MSTConfig: a=0, b=-0.5, c=0.5": np.zeros(len(facet_dirs)),
    "MSTConfig: a=0, b=0.5, c=-0.5": np.zeros(len(facet_dirs)),
    "MSTConfig: a=-0.3333333333333333, b=-0.3333333333333333, c=-0.3333333333333333": np.zeros(len(facet_dirs)),
    "MSTConfig: a=0.3333333333333333, b=0.3333333333333333, c=0.3333333333333333": np.zeros(len(facet_dirs)),
    "MSTConfig: a=-0.3333333333333333, b=-0.3333333333333333, c=0.3333333333333333": np.zeros(len(facet_dirs)),
    "MSTConfig: a=-0.3333333333333333, b=0.3333333333333333, c=-0.3333333333333333": np.zeros(len(facet_dirs)),
    "MSTConfig: a=0.3333333333333333, b=-0.3333333333333333, c=-0.3333333333333333": np.zeros(len(facet_dirs)),
    "MSTConfig: a=0.3333333333333333, b=0.3333333333333333, c=-0.3333333333333333": np.zeros(len(facet_dirs)),
    "MSTConfig: a=0.3333333333333333, b=-0.3333333333333333, c=0.3333333333333333": np.zeros(len(facet_dirs)),
    "MSTConfig: a=-0.3333333333333333, b=0.3333333333333333, c=0.3333333333333333": np.zeros(len(facet_dirs)),
    "Dfs": np.zeros(len(facet_dirs)),
    "Bfs": np.zeros(len(facet_dirs)),
    "RandomSpanningTree": np.zeros(len(facet_dirs)),
}


for i, facet_dir in enumerate(facet_dirs):
    with open(DATASET_DIR / facet_dir / "heuristic_results.json") as f:
        data = json.load(f)

        for j in range(1000):
            results = data[f"polytope_{j}"]
            for key in heur.keys():
                heur[key][i] += int(results[key])

for key in heur.keys():
    heur[key] /= 1000


random_rates = heur["RandomSpanningTree"]
threshold = 0.02  # within 2% = similar

better, similar, worse = {}, {}, {}

for label, rates in heur.items():
    if label == "RandomSpanningTree":
        continue
    avg_diff = np.mean(rates - random_rates)
    if avg_diff > threshold:
        better[label] = rates
    elif avg_diff < -threshold:
        worse[label] = rates
    else:
        similar[label] = rates

print(better)
print(similar)
print(worse)
# add random to all plots as bold reference line
for d in [better, similar, worse]:
    d["RandomSpanningTree"] = random_rates

x_labels = [f"{i}-{i+4}" for i in range(5, 100, 5)]

x = np.arange(len(x_labels))
def make_plot(data, title, filename):
    colors = plt.cm.hsv(np.linspace(0, 1, len(data)))
    fig, ax = plt.subplots(figsize=(14, 6))
    for (label, rates), color in zip(data.items(), colors):
        lw = 3 if label == "RandomSpanningTree" else 1
        ax.plot(x, rates, marker='o', markersize=3, label=label, color=color, linewidth=lw)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.set_xlabel("Facet count bin")
    ax.set_ylabel("Success rate")
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, format="svg")
    plt.show()

make_plot(better, "Better than random", "better_than_random.svg")
make_plot(similar, "Similar to random", "similar_to_random.svg")
make_plot(worse, "Worse than random", "worse_than_random.svg")
# x_labels = [f"{i}-{i+4}" for i in range(5, 100, 5)]
#
# fig, ax = plt.subplots(figsize=(14, 6))
#
# for label, rates in heur.items():
#     ax.plot(x, rates, marker='o', markersize=3, label=label)
#
# ax.set_xticks(x)
# ax.set_xticklabels(x_labels, rotation=45, ha='right')
# ax.set_xlabel("Facet count bin")
# ax.set_ylabel("Success rate")
# ax.set_title("PriorityTraversal success rate per facet bin")
# ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
# ax.set_ylim(0, 1)
# ax.grid(True, alpha=0.3)
#
# plt.tight_layout()
# plt.savefig("priority_traversal_success_rates.png", dpi=150)
# plt.show()

random_rates = heur["RandomSpanningTree"]

avg_diffs = {
    label: np.mean(rates - random_rates)
    for label, rates in heur.items()
    if label != "RandomSpanningTree"
}

sorted_diffs = sorted(avg_diffs.items(), key=lambda x: x[1])

top5_worse = dict(sorted_diffs[:3])
top5_better = dict(sorted_diffs[-3:])

plot_data = {}
plot_data.update(top5_worse)
plot_data["RandomSpanningTree"] = random_rates
plot_data.update(top5_better)

colors = plt.cm.tab20(np.linspace(0, 1, len(plot_data)))
fig, ax = plt.subplots(figsize=(14, 6))

for (label, rates), color in zip(plot_data.items(), colors):
    lw = 3 if label == "RandomSpanningTree" else 1
    ax.plot(x, heur[label] if label != "RandomSpanningTree" else random_rates,
            marker='o', markersize=3, label=label, color=color, linewidth=lw)

ax.set_xticks(x)
ax.set_xticklabels(x_labels, rotation=45, ha='right')
ax.set_xlabel("Facet count bin")
ax.set_ylabel("Success rate")
ax.set_title("Top 3 best, top 3 worst vs random")
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("top3.svg", format="svg")
plt.show()
