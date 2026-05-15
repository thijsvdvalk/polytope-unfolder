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
N_CORES = 6

def process_polytope(args):
    facet_dir, i = args
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

def process_facet_dir_single(facet_dir, i):
    output_path = DATASET_DIR / facet_dir / f"polytope_{i}_estimate.json"
    if output_path.exists():
        return  # skip already computed
    
    _, data = process_polytope((facet_dir, i))
    
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

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
    
    interleaved_dirs = [f"{i}-{i+4}_facets" for i in range(25, 50, 5)]  # 25-99

    # Process early dirs normally

    # Process interleaved dirs: polytope 0 from all, then polytope 1 from all, etc.
    for i in range(1000):
        args = [(facet_dir, i) for facet_dir in interleaved_dirs]
        with Pool(processes=N_CORES) as pool:
            results = pool.map(process_polytope, args)
        
        for facet_dir, (_, data) in zip(interleaved_dirs, results):
            output_path = DATASET_DIR / facet_dir / f"polytope_{i}_estimate.json"
            with open(output_path, "w") as f:
                json.dump(data, f, indent=2)
        
