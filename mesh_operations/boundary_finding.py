import numpy as np
from scipy.spatial import Delaunay

def triangulate_and_find_boundaries(faces, vertices):
    """
    Perform Delaunay triangulation and identify the boundary vertices of the inner and outer regions.

    Args:
    - vertices: An array of vertices for the mesh.

    Returns:
    - delaunay: The Delaunay triangulation object.
    - outer_boundary_vertices: The vertices on the outer boundary.
    - inner_boundary_vertices: The vertices on the inner boundary.
    """

    # Find boundary edges (edges that appear exactly once)
    edges = {}
    for face in faces:
        for i in range(3):
            edge = tuple(sorted([face[i], face[(i + 1) % 3]]))
            if edge in edges:
                edges[edge] += 1
            else:
                edges[edge] = 1

    boundary_edges = [edge for edge, count in edges.items() if count == 1]
    boundary_vertices = list(set([vertex for edge in boundary_edges for vertex in edge]))

    # Assuming the inner boundary has vertices with higher indices due to the generation method
    delaunay = Delaunay(vertices)
    outer_boundary_vertices = [vertex for vertex in boundary_vertices if vertex in delaunay.convex_hull.flatten()]
    inner_boundary_vertices = list(set(boundary_vertices) - set(outer_boundary_vertices))

    return faces, outer_boundary_vertices, inner_boundary_vertices