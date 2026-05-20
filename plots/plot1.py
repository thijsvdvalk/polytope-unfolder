"""

1. Relation between facet count and hardness of a polytope.

2. Comparison of the algorithms of course. So we need to get like things to show.
    - Baseline we have.
    - We have either true or false for each polytope.
    - We can group polytopes on their hardness, and calculate avg per algorithm to see how they perform, maybe some group stays the same even for harder polytopes.
    - so on x-axis can have hardness, y-axis success rate, and have 3 lines for each algorithm.
    - But then for each hardness I must have the same amount of samples.

"""

from pathlib import Path
import matplotlib.pyplot as plt
import json
import numpy as np


DATASET_DIR = Path("../polytopes_dataset") / "normal_distribution"
facet_dirs = [f"{i}-{i+4}_facets" for i in range(5, 70, 5)]

for facet_dir in facet_dirs:
    with open(DATASET_DIR / facet_dir / "overlap_free_estimates.json", "r") as f:
        s = json.load(f)
        avg = np.sum([val["p"] for val in s.values()])  / 1000
        print(avg)
