from numpy.typing import NDArray
from polytope_core.builder import PolytopeBuilder
from polytope_core.polytope import Polytope,  Net
import numpy as np
import networkx as nx


poly = PolytopeBuilder.simplex4()
traversal = [(0,1), (1,2), (2,3), (3,4)]
net = poly.unfold(traversal)
print(net.overlaps())
