import meshio
from shadowspy.shape import get_centroids as get_cents, get_surface_normals

def import_mesh(mesh_path, get_normals=False, get_centroids=False):
    # use meshio to import obj shapefile
    mesh = meshio.read(
        filename=mesh_path,  # string, os.PathLike, or a buffer/open file
    )

    V = mesh.points
    # V = V.astype(np.float32)  # embree is anyway single precision # destroys normals
    V = V[:, :3]
    F = mesh.cells[0].data

    if (not get_normals) and (not get_centroids):
        return V, F

    if get_normals:
        P = get_cents(V, F)
        N = get_surface_normals(V, F)
        N[(N * P).sum(1) < 0] *= -1
        if get_centroids:
            return V, F, N, P
        else:
            return V, F, N
