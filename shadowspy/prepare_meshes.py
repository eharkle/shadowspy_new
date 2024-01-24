import logging
import time

import meshio
import numpy as np
import rioxarray as rio

from shadowspy.coord_tools import unproject_stereographic, sph2cart
from shadowspy.mesh_util import get_uniform_triangle_mesh


def make(base_resolution, decimation_rates, tif_path, out_path, mesh_ext='.xmf', plarad=1737.4, lonlat0=(0, -90)):

    rds = rio.open_rasterio(tif_path)
    tiff_resolution = int(rds.rio.resolution()[0])

    # print(abs(rds.rio.resolution()[0]), abs(rds.rio.resolution()[1]))
    assert abs(abs(rds.rio.resolution()[0]) - abs(rds.rio.resolution()[1])) < 1.e-16

    # if the required dn1 (base resolution) is larger than the native GTiff one, decimate the input
    if base_resolution == tiff_resolution:
        # xgrid = rds.coords['y'].values*-1.e-3
        # ygrid = rds.coords['x'].values*1.e-3
        # dem = rds.data[0].T[:, ::-1] * 1.e-3
        xgrid = rds.coords['x'].values * 1.e-3
        ygrid = rds.coords['y'].values * 1.e-3
        dem = rds.data[0][:, ::] * 1.e-3
    elif base_resolution > tiff_resolution and np.mod(base_resolution, tiff_resolution) == 0:
        decimation = int(base_resolution / tiff_resolution)
        xgrid = rds.coords['x'].values[::decimation] * 1.e-3
        ygrid = rds.coords['y'].values[::decimation] * 1.e-3
        dem = rds.data[0][::decimation, ::decimation] * 1.e-3
    else:
        logging.error(f"* Requested b{base_resolution} < tiff resolution ({tiff_resolution}) or"
                      f"not a multiple.")
        exit()

    logging.debug(f"- GTiff read at {base_resolution}mpp (from original {tiff_resolution}mpp).")

    mesh_versions = {}
    for decimation in decimation_rates:
        start = time.time()
        mesh_versions[decimation] = get_uniform_triangle_mesh(xgrid, ygrid, dem, decimation=decimation)
        logging.debug(
            f"- Mesh of shape {mesh_versions[decimation]['shape']} (decimation={decimation}) set up "
            f"after {round(time.time()-start, 5) * 1e3} milli-seconds")

        for coord_style in ["stereo", "cart"]:

            if coord_style == "cart":

                lon, lat = unproject_stereographic(mesh_versions[decimation]['V'][:, 0],
                                                   mesh_versions[decimation]['V'][:, 1], lonlat0[0], lonlat0[1],
                                                   R=plarad) # + mesh_versions[decimation]['V'][:, 2])

                x, y, z = sph2cart(plarad + mesh_versions[decimation]['V'][:, 2], lat, lon)
                V_cart = np.vstack([x, y, z]).T
                mesh = meshio.Mesh(V_cart, [('triangle', mesh_versions[decimation]['F'])])
                # fout = f"in/shackleton_{np.product(mesh_versions[decimation]['shape'])*2}.ply"
                fout = f"{out_path}b{base_resolution}_dn{decimation}{mesh_ext}"
            else:
                mesh = meshio.Mesh(mesh_versions[decimation]['V'], [('triangle', mesh_versions[decimation]['F'])])
                # fout = f"in/shackleton_{np.product(mesh_versions[decimation]['shape'])*2}_st.ply"
                fout = f"{out_path}b{base_resolution}_dn{decimation}_st{mesh_ext}"

            mesh.write(fout)
            print(f"- Delauney mesh computed and saved to {fout}.")
            logging.debug(f"- Delauney mesh computed and saved to {fout}.")

    logging.debug(f"(decimation,num_faces):\n{[(dec, len(mesh['F'])) for dec, mesh in mesh_versions.items()]}")

    return fout

if __name__ == '__main__':
    base_resolution = 30
    decimation_rates = [1]  # , 2, 4, 10, 20]
    path = ''
    tif_path = path + '.tif'

    make(base_resolution, decimation_rates, tif_path, "in/")
