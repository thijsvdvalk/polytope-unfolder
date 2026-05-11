from polytope_core.builder import PolytopeBuilder as pb
from polytope_core.polytope import Polytope
from polytope_core.tet_plotter import plot_tets
import numpy as np
from polytope_core.overlap import overlaps
polytope = pb.deserialize("./polytopes/failing.json")
print(polytope.neigh_graph.nodes)
traversal_1=[(1, 7), (7, 4), (4, 3), (1, 0), (4, 5), (0, 6), (6, 8), (5, 2)] # -> overlaps=False
traversal_2=[(4, 3), (4, 5), (4, 7), (5, 2), (7, 1), (1, 0), (0, 6), (6, 8)] # -> overlaps=True

net1 = polytope.unfold(traversal_1)
net2 = polytope.unfold(traversal_2)

print(net1.overlaps())
print(net2.overlaps())


tet_1 = np.array([
    [-1.52781014,  0.06482162,  2.5883421 ],
    [-0.42120269,  0.23939833,  3.20497405],
    [-0.76066868, -1.89400814,  0.79775459],
    [ 2.14422405,  2.48484855,  0.79471703]
])
tet_2 = np.array([
    [ 0.,          0.,          0.        ],
    [-1.52781014,  0.06482162,  2.5883421 ],
    [-0.76066868, -1.89400814,  0.79775459],
    [ 2.14422405,  2.48484855,  0.79471703]
])


print(f"{overlaps(tet_1, tet_2)=}")
plot_tets(tet_1, tet_2)
