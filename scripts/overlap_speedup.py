from polytope_core.builder import PolytopeBuilder as pb
import networkx as nx
from heur_algo_imps import RandomSpanningTree as rst
from pathlib import Path
import time

dataset_path = Path("..") / "polytopes_dataset" / "normal_distribution" / "45-49_facets"

for i in range(1000):
    polytope = pb.deserialize(dataset_path / f"polytope_{i}.json")
    st = rst.spanning_tree(polytope)

    start = time.perf_counter()
    overlap = polytope.unfold_from_spanning_tree(st).overlaps()
    end = time.perf_counter()
    took = end - start
    

    start2 = time.perf_counter()
    ov2 = polytope.unfold_from_spanning_tree_short_circuit(st)
    end2 = time.perf_counter()
    took2 = end2 - start2

    print(overlap)
    print(ov2)
    print(took / took2)
    print()
