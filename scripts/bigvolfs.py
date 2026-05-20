from heur_algo_imps import *


from pathlib import Path
from polytope_core.builder import PolytopeBuilder as pb
from polytope_core.heur_algo_abc import PriorityTraversalConfig, MSTConfig

DATASET_DIR = Path("../polytopes_dataset") / "normal_distribution"
for i in range(1):
    polytope = pb.deserialize(DATASET_DIR / "40-44_facets" / f"polytope_{i}.json")
    polytope.init_node_weigths()
    st = PriorityTraversal.spanning_tree(polytope, PriorityTraversalConfig(0.5, 0.5, False))
    polytope.overlap_free_unfolding(st)

    #tree = polytope.overlap_free_unfolding(BigAnglefs.spanning_tree(polytope))
    # print(tree)



