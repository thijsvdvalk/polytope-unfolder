from polytope_core.polytope import Polytope
from polytope_core.builder import PolytopeBuilder as pb
import networkx as nx
from hypothesis import given, settings, strategies as st
import random


def brute_force_all_net(polytope: Polytope):
    """Brute forces every spanning tree to test for all-net property."""
    for tree in nx.SpanningTreeIterator(polytope.neigh_graph):
        assert not polytope.unfold_from_spanning_tree(tree).overlaps()


# def test_orthoplex_all_net_brute_force():
#     orthoplex = pb.orthoplex4()
#     brute_force_all_net(orthoplex)


def test_simplex_all_net_brute_force():
    """Brute forces the simplex-4 to test for the proven all-net property."""
    simplex = pb.simplex4()
    brute_force_all_net(simplex)


def test_orthoplex_all_net_sampled():
    """Samples random spanning trees for the orthoplex-4 and asserts no overlap because the orthoplex-4 is all-net."""
    orthoplex = pb.orthoplex4()
    for _ in range(100):
        st = nx.random_spanning_tree(orthoplex.neigh_graph)
        assert not orthoplex.unfold_from_spanning_tree(st).overlaps()


@given(st.integers(min_value=5, max_value=50))
@settings(max_examples=10, deadline=None)
def test_overlap_invariant_to_traversal_order(num_points):
    """Any traversal order of the same spanning tree should yield the same result for overlap."""
    polytope = pb.random_normal(num_points)
    sp = nx.random_spanning_tree(polytope.neigh_graph)

    results = set()
    for _ in range(10):
        traversal = random_traversal_sp(sp)
        net = polytope._unfold(traversal)
        result = net.overlaps()
        results.add(result)

    assert len(results) == 1, (
        "Different traversals of the same spanning tree disagree on overlap."
    )


def random_traversal_sp(sp: nx.Graph) -> list[tuple[int, int]]:
    """Given a spanning tree produces a random traversal."""
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
