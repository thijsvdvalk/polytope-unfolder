from itertools import combinations
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np


def _tet_faces(verts):
    # all combinations of 3 vertices from 4 = the 4 triangular faces
    return [verts[list(f)] for f in combinations(range(4), 3)]

def plot_tets(*tets):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = matplotlib.cm.tab10.colors  # 10 distinct colors

    for verts, color in zip(tets, colors):
        faces = _tet_faces(verts)
        poly = Poly3DCollection(faces, alpha=0.3, facecolor=color, edgecolor='black')
        ax.add_collection3d(poly)

    all_verts = np.vstack(tets)
    for setter, coord in zip([ax.set_xlim, ax.set_ylim, ax.set_zlim], all_verts.T):
        setter(coord.min(), coord.max())

    ax.set_aspect('equal')

    plt.show()


