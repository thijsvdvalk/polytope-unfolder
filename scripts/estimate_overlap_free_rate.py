from pathlib import Path
from polytope_core.builder import PolytopeBuilder as pb
from heur_algo_imps import RandomSpanningTree as rst
from statsmodels.stats.proportion import proportion_confint
from multiprocessing import Pool
import json
import random
import numpy as np
import time

SEED = 4444
SAMPLES_PER_POLYTOPE = 385
DATASET_DIR = Path("../polytopes_dataset") / "normal_distribution"
N_CORES = 10


def process_polytope(args):
    facet_dir, i = args
    random.seed(SEED + i)
    np.random.seed(SEED + i)
    polytope = pb.deserialize(DATASET_DIR / facet_dir / f"polytope_{i}.json")

    successes = 0
    for _ in range(SAMPLES_PER_POLYTOPE):
        successes += int(polytope.overlap_free_unfolding(rst.spanning_tree(polytope)))

    success_rate = successes / SAMPLES_PER_POLYTOPE
    low, high = proportion_confint(successes, SAMPLES_PER_POLYTOPE, alpha=0.05, method='wilson')

    return i, {"p": success_rate, "conf_int_95": [low, high]}


def load_results(facet_dir) -> dict:
    path = DATASET_DIR / facet_dir / "overlap_free_estimates.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save_results(facet_dir, results: dict):
    with open(DATASET_DIR / facet_dir / "overlap_free_estimates.json", "w") as f:
        json.dump(results, f, indent=2)


def process_facet_dir(facet_dir):
    existing = load_results(facet_dir)
    remaining = [i for i in range(1000) if f"polytope_{i}" not in existing]

    if not remaining:
        print(f"[{facet_dir}] Already complete, skipping.")
        return

    print(f"[{facet_dir}] Starting — {len(remaining)} remaining, {len(existing)} already done.")

    args = [(facet_dir, i) for i in remaining]
    total = len(args)
    done = 0
    start_time = time.perf_counter()

    results = dict(existing)

    with Pool(processes=N_CORES) as pool:
        for i, data in pool.imap_unordered(process_polytope, args):
            results[f"polytope_{i}"] = data
            done += 1
            save_results(facet_dir, results)

            elapsed = time.perf_counter() - start_time
            rate = done / elapsed
            remaining_count = total - done
            eta = remaining_count / rate if rate > 0 else float('inf')
            print(
                f"[{facet_dir}] {done}/{total} done | "
                f"elapsed: {elapsed:.1f}s | "
                f"rate: {rate:.1f}/s | "
                f"ETA: {eta:.1f}s"
            )

    print(f"[{facet_dir}] Finished in {time.perf_counter() - start_time:.1f}s")


if __name__ == "__main__":
    facet_dirs = [f"{i}-{i+4}_facets" for i in range(5, 100, 5)]
    for facet_dir in facet_dirs:
        process_facet_dir(facet_dir)
