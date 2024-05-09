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



def triangulate_and_find_boundaries_vectorized(faces, vertices):
    """
    Perform Delaunay triangulation and identify the boundary vertices of the inner and outer regions.

    Args:
    - vertices: An array of vertices for the mesh.

    Returns:
    - faces: The input faces of the mesh.
    - outer_boundary_vertices: The vertices on the outer boundary.
    - inner_boundary_vertices: The vertices on the inner boundary.
    """

    # Create edges
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = np.sort(edges, axis=1)  # Sort edges to make identical ones adjacent
    edges, counts = np.unique(edges, axis=0, return_counts=True)

    # Boundary edges are those that appear exactly once
    boundary_edges = edges[counts == 1]
    boundary_vertices = np.unique(boundary_edges)

    # Delaunay triangulation and identification of convex hull vertices
    delaunay = Delaunay(vertices) # taking 80% of the whole merge_inout
    convex_hull_vertices = np.unique(delaunay.convex_hull)

    # Identify outer and inner boundary vertices based on convex hull
    outer_boundary_vertices = np.intersect1d(boundary_vertices, convex_hull_vertices)
    inner_boundary_vertices = np.setdiff1d(boundary_vertices, outer_boundary_vertices)

    return faces, outer_boundary_vertices, inner_boundary_vertices
