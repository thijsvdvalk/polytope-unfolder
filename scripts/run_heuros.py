from heur_algo_imps import *
from pathlib import Path
from polytope_core.builder import PolytopeBuilder as pb
from polytope_core.heur_algo_abc import PriorityTraversalConfig, MSTConfig
import json
import multiprocessing as mp

# --- configs ---
priority_configs = [
    PriorityTraversalConfig(-1, 0),
    PriorityTraversalConfig(1, 0),
    PriorityTraversalConfig(0, -1),
    PriorityTraversalConfig(0, 1),

    PriorityTraversalConfig(-0.5, -0.5),
    PriorityTraversalConfig(0.5, 0.5),
    PriorityTraversalConfig(0.5, -0.5),
    PriorityTraversalConfig(-0.5, 0.5),
]

mst_configs = [
    # single property
    MSTConfig(-1, 0, 0),   # max dihedral angle
    MSTConfig(1, 0, 0),    # min dihedral angle
    MSTConfig(0, -1, 0),   # max shared face area
    MSTConfig(0, 1, 0),    # min shared face area
    MSTConfig(0, 0, -1),   # max centroids distance
    MSTConfig(0, 0, 1),    # min centroids distance

    # dihedral + area
    MSTConfig(-0.5, -0.5, 0),  # max both
    MSTConfig(0.5, 0.5, 0),    # min both
    MSTConfig(-0.5, 0.5, 0),   # max dihedral, min area
    MSTConfig(0.5, -0.5, 0),   # min dihedral, max area

    # dihedral + centroid
    MSTConfig(-0.5, 0, -0.5),  # max both
    MSTConfig(0.5, 0, 0.5),    # min both
    MSTConfig(-0.5, 0, 0.5),   # max dihedral, min centroid
    MSTConfig(0.5, 0, -0.5),   # min dihedral, max centroid

    # area + centroid
    MSTConfig(0, -0.5, -0.5),  # max both
    MSTConfig(0, 0.5, 0.5),    # min both
    MSTConfig(0, -0.5, 0.5),   # max area, min centroid
    MSTConfig(0, 0.5, -0.5),   # min area, max centroid

    # all three
    MSTConfig(-1/3, -1/3, -1/3),  # max all
    MSTConfig(1/3, 1/3, 1/3),     # min all
    MSTConfig(-1/3, -1/3, 1/3),   # max dihedral, max area, min centroid
    MSTConfig(-1/3, 1/3, -1/3),   # max dihedral, min area, max centroid
    MSTConfig(1/3, -1/3, -1/3),   # min dihedral, max area, max centroid
    MSTConfig(1/3, 1/3, -1/3),    # min dihedral, min area, max centroid
    MSTConfig(1/3, -1/3, 1/3),    # min dihedral, max area, min centroid
    MSTConfig(-1/3, 1/3, 1/3),    # max dihedral, min area, min centroid
]

traversal_configs = [
    Dfs,
    Bfs,
    RandomSpanningTree
]

DATASET_DIR = Path("../polytopes_dataset")
facet_dirs = [f"{2**i}-{(2**(i+1)) - 1}_facets" for i in range(2, 11)]
distributions = ['normal', 'uniform', 'exponential', 'uniform_on_hypersphere']

def get_output_file(facet_dir: str, distribution) -> Path:
    return DATASET_DIR / distribution / facet_dir / "heuristic_results.json"

def load_results(facet_dir: str, distribution: str) -> dict:
    output_file = get_output_file(facet_dir, distribution)
    if not output_file.exists():
        return {}
    try:
        with open(output_file) as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}

def save_results(facet_dir: str, distribution: str, results: dict):
    output_file = get_output_file(facet_dir, distribution)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

def process_batch(args):
    facet_dir, distribution, batch_start, lock = args
    batch_results = {}
    with lock:
        existing = load_results(facet_dir, distribution)

    for i in range(batch_start, batch_start + 100):
        polytope_key = f"polytope_{i}"

        if polytope_key in existing:
            continue

        try:
            polytope = pb.deserialize(DATASET_DIR / distribution / facet_dir / f"polytope_{i}.json")
            polytope.init_node_weigths()
            polytope.init_edge_weigths()
        except Exception as e:
            print(f"Failed to load {facet_dir}/{polytope_key}: {e}")
            continue

        polytope_results = {}

        for config in priority_configs:
            try:
                st = PriorityTraversal.spanning_tree(polytope, config)
                polytope_results[str(config)] = bool(polytope.overlap_free_unfolding(st))
            except Exception as e:
                print(e)
                polytope_results[str(config)] = None

        for config in mst_configs:
            try:
                st = MinSpanningTree.spanning_tree(polytope, config)
                polytope_results[str(config)] = bool(polytope.overlap_free_unfolding(st))
            except Exception as e:
                print(e)
                polytope_results[str(config)] = None

        for cls in traversal_configs:
            try:
                st = cls.spanning_tree(polytope, None)
                polytope_results[cls.__name__] = bool(polytope.overlap_free_unfolding(st))
            except Exception as e:
                print(e)
                polytope_results[cls.__name__] = None

        batch_results[polytope_key] = polytope_results

    # write all 100 at once
    with lock:
        results = load_results(facet_dir, distribution)
        results.update(batch_results)
        save_results(facet_dir, distribution, results)

    print(f"Done batch: {facet_dir} polytopes {batch_start}-{batch_start+99}")


if __name__ == "__main__":
    print("heii")
    tasks = [
        (facet_dir, distribution, batch_start)
        for facet_dir in facet_dirs
        for distribution in distributions
        for batch_start in range(0, 500, 100)
    ]

    manager = mp.Manager()
    lock = manager.Lock()
    tasks = [(fd, dis, bs, lock) for fd, dis, bs in tasks]

    print(f"{len(tasks)} batches to process")

    num_cores = 10
    with mp.Pool(processes=num_cores) as pool:
        pool.map(process_batch, tasks)
