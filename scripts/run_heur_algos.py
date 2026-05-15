# from pathlib import Path
# from polytope_core.builder import PolytopeBuilder as pb
# import json
# from heur_algo_imps import *
#
# algo = Dfs
#
# facet_dirs = [f"{i}-{i+4}_facets" for i in range(5, 100, 5)] # 30 should be 100
# dataset_dir = Path("../polytopes_dataset") / "normal_distribution"
#
# for facet_dir in facet_dirs:
#     overlap_frees = []
#     for i in range(1000):
#         polytope = pb.deserialize(dataset_dir / facet_dir / f"polytope_{i}.json")
#         net = polytope.unfold_from_spanning_tree(algo.spanning_tree(polytope))
#         overlap_frees.append(not net.overlaps())
#
#
#
#     with open(dataset_dir / facet_dir / f"{algo.__name__}.json", "w") as f:
#         json.dump(overlap_frees, f, indent=2)
#
#     print(f"Finished for {facet_dir}")
from pathlib import Path
from polytope_core.builder import PolytopeBuilder as pb
from multiprocessing import Pool
import json
from heur_algo_imps import Dfs, Bfs

DATASET_DIR = Path("../polytopes_dataset") / "normal_distribution"
N_CORES = 4

def process_polytope(args):
    facet_dir, i, algo = args
    polytope = pb.deserialize(DATASET_DIR / facet_dir / f"polytope_{i}.json")
    net = polytope.unfold_from_spanning_tree(algo.spanning_tree(polytope))
    return i, not net.overlaps()

def process_facet_dir(facet_dir, algo):
    args = [(facet_dir, i, algo) for i in range(1000)]
    with Pool(processes=N_CORES) as pool:
        results = pool.map(process_polytope, args)
    
    results.sort(key=lambda x: x[0])
    overlap_frees = [r for _, r in results]
    
    with open(DATASET_DIR / facet_dir / f"{algo.__name__}.json", "w") as f:
        json.dump(overlap_frees, f, indent=2)
    print(f"Finished {algo.__name__} for {facet_dir}")

if __name__ == "__main__":
    facet_dirs = [f"{i}-{i+4}_facets" for i in range(5, 100, 5)]
    for algo in [Dfs, Bfs]:
        for facet_dir in facet_dirs:
            process_facet_dir(facet_dir, algo)
