from heur_algo_imps import Dfs, Bfs, RandomWalk
from polytope_core.builder import PolytopeBuilder as pb
from polytope_core.polytope import Polytope

simplex = pb.simplex4()
st = RandomWalk.spanning_tree(simplex)
net = simplex.unfold_from_spanning_tree(st)
print(net.overlaps())
