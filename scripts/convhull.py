from numpy.typing import NDArray
from polytope_core.polytope import Polytope, Net
import numpy as np
import networkx as nx
from polytope_core.tet_plotter import plot_tets

poly = PolytopeBuilder.simplex4()
traversal = [(0,1), (1,2), (2,3), (3,4)]
