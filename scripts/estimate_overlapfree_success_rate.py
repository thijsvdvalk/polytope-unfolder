# from pathlib import Path
# from polytope_core.builder import PolytopeBuilder as pb
# from heur_algo_imps import RandomSpanningTree as rst
# from statsmodels.stats.proportion import proportion_confint
# import json
# import random
# import numpy as np
#
# SEED = 4444
# random.seed(SEED)
# np.random.seed(SEED)
#
# facet_dirs = [f"{i}-{i+4}_facets" for i in range(5, 30, 5)] # 30 should be 100
# dataset_dir = Path("../polytopes_dataset") / "normal_distribution"
#
# for facet_dir in facet_dirs:
#     print(f"Started for {facet_dir}")
#     success_estimates = []
#     samples_per_polytope = 385
#
#     for i in range(1000):
#         successes = 0
#         for _ in range(samples_per_polytope):
#             polytope = pb.deserialize(dataset_dir / facet_dir / f"polytope_{i}.json")
#             net = polytope.unfold_from_spanning_tree(rst.spanning_tree(polytope))
#             successes += int(not net.overlaps())
#
#         success_rate = successes / samples_per_polytope
#         low, high = proportion_confint(successes, samples_per_polytope, alpha=0.05, method='wilson')
#         success_estimates.append({f"polytope_{i}": {"p": success_rate, "conf_int_95": [low, high]}})
#
#         print(f"\tdone for {i}")
#
#     with open(dataset_dir / facet_dir / "overlap_free_estimates.json", "w") as f:
#         json.dump(success_estimates, f, indent=2)
#
#     print(f"Finished for {facet_dir}")
from pathlib import Path
from polytope_core.builder import PolytopeBuilder as pb
from heur_algo_imps import RandomSpanningTree as rst
from statsmodels.stats.proportion import proportion_confint
from multiprocessing import Pool
import json
import random
import numpy as np

SEED = 4444
SAMPLES_PER_POLYTOPE = 385
DATASET_DIR = Path("../polytopes_dataset") / "normal_distribution"
N_CORES = 12

def process_polytope(args):
    facet_dir, i = args
    # each process needs its own seed to avoid identical random sequences
    random.seed(SEED + i)
    np.random.seed(SEED + i)

    polytope = pb.deserialize(DATASET_DIR / facet_dir / f"polytope_{i}.json")
    
    successes = 0
    for _ in range(SAMPLES_PER_POLYTOPE):
        net = polytope.unfold_from_spanning_tree(rst.spanning_tree(polytope))
        successes += int(not net.overlaps())
    
    success_rate = successes / SAMPLES_PER_POLYTOPE
    low, high = proportion_confint(successes, SAMPLES_PER_POLYTOPE, alpha=0.05, method='wilson')
    
    return i, {"p": success_rate, "conf_int_95": [low, high]}

def process_facet_dir(facet_dir):
    print(f"Started for {facet_dir}")
    
    args = [(facet_dir, i) for i in range(1000)]
    with Pool(processes=N_CORES) as pool:
        results = pool.map(process_polytope, args)
    
    success_estimates = {f"polytope_{i}": data for i, data in results}
    
    with open(DATASET_DIR / facet_dir / "overlap_free_estimates.json", "w") as f:
        json.dump(success_estimates, f, indent=2)
    
    print(f"Finished for {facet_dir}")



if __name__ == "__main__":
    facet_dirs = [f"{i}-{i+4}_facets" for i in range(10, 100, 5)]
    for facet_dir in facet_dirs:
        process_facet_dir(facet_dir)
