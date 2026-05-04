from numpy.typing import NDArray
from polytope_core.polytope import Polytope, PolytopeBuilder, Net
import numpy as np
import networkx as nx


poly = PolytopeBuilder.simplex4()
poly2 = Polytope.deserialize("./polytopes/rand10.json")
traversal = [(0,1), (1,2), (2,3), (3,4)]
nodes = nx.dfs_edges(poly2.neigh_graph, source=0)
nodes_trav = [(x, y) for x, y in nodes]
net = Net(poly, traversal)
# print(net.overlaps())
net2 = Net(poly2, nodes_trav)
print(net2.overlaps())
# todo
# make code nicer
# test code
