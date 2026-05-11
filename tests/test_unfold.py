from polytope_core.polytope import Polytope
from polytope_core.builder import PolytopeBuilder as pb
from itertools import combinations
import networkx as nx
from hypothesis import given, settings, strategies as st
from polytope_core.builder import PolytopeBuilder
import numpy as np
import random

# although many traversals are the same here, we just test them all to see if the code actuall works
# def test_simplex_all_net_brute_force():
#     simplex = pb.simplex4()
#     neigh_graph = simplex.neigh_graph 
#     spanning_trees = []
#
#     for edges in combinations(neigh_graph.edges(), 4):
#         possible_st = nx.Graph()
#         possible_st.add_nodes_from(neigh_graph.nodes)
#         possible_st.add_edges_from(edges)
#         if nx.is_connected(possible_st):
#             for source in neigh_graph.nodes:
#                 traversal = list(nx.bfs_edges(possible_st, source=source)) # this just makes sure spanning tree is in traversal format
#                 spanning_trees.append(traversal)
#
#     for mst in spanning_trees:
#         net = simplex.unfold(mst)
#         assert not net.overlaps()
#
#
# def test_orthoplex_all_net_sampled():
#     orthoplex = pb.orthoplex4()
#     G = orthoplex.neigh_graph
#     for node in G.nodes:
#         traversal_bfs = list(nx.bfs_edges(G, source=node))
#         traversal_dfs = list(nx.dfs_edges(G, source=node))
#         assert not orthoplex.unfold(traversal_bfs).overlaps()
#         assert not orthoplex.unfold(traversal_dfs).overlaps()


# create random polytope and random spanning tree, see that the overlap is the same no matter what traversal of the spanning tree.
@given(st.integers(min_value=5, max_value=25))
@settings(max_examples=10, deadline=None)
def test_b(num_points):
    polytope = pb.random(num_points)
    sp = nx.random_spanning_tree(polytope.neigh_graph)
    
    results = []
    for _ in range(10):
        traversal = random_traversal_sp(sp)
        net = polytope.unfold(traversal)
        result = net.overlaps()
        results.append((traversal, result))
    
    values = set(r for _, r in results)
    if len(values) != 1:
        pb.serialize(polytope, "polytopes/failing.json")
        print(f"\nnum_points={num_points}")
        for traversal, result in results:
            print(f"  traversal={traversal} -> overlaps={result}")
        assert False, f"traversals disagree: got both True and False"

def random_traversal_sp(sp: nx.Graph) -> list[tuple[int, int]]:
    start_node = random.choice(list(sp.nodes))
    traversal = []
    visited = {start_node}
    queue = [(start_node, n) for n in list(sp.neighbors(start_node))]
    while queue:
        random.shuffle(queue)
        prev, cur = queue.pop(0)
        if cur in visited:
            continue
        visited.add(cur)
        traversal.append((prev, cur))
        neighbors = [(cur, n) for n in sp.neighbors(cur) if n not in visited]
        queue.extend(neighbors)

    return traversal
