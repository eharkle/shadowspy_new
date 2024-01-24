import glob
import os
import shutil
import time
import pandas as pd

from examples.download_kernels import download_kernels
from shadowspy import prepare_meshes
from shadowspy.render_dem import render_at_date
import xarray as xr

if __name__ == '__main__':

    # compute direct flux from the Sun
    Fsun = 1361  # W/m2
    Rb = 1737.4 # km
    base_resolution = 120
    root = "examples/"
    os.makedirs(root, exist_ok=True)

    # download kernels
    download_kernels()

    # Elevation/DEM GTiff input
    indir = f"{root}aux/"
    tif_path = f'{indir}dm2_120mpp.tif'#
    meshpath = tif_path.split('.')[0]
    fartopo_path = f"{indir}dm2_basemap_120mpp.tif"
    fartopomesh = fartopo_path.split('.')[0]

    # prepare mesh of the input dem
    start = time.time()
    print(f"- Computing trimesh for {tif_path}...")

    # regular delauney mesh
    ext = '.vtk'
    fartopo = xr.load_dataset(tif_path)
    fartopo -= Rb*1e3
    fartopo.band_data.rio.to_raster(f"{indir}dm2_rescaled.tif")
    prepare_meshes.make(base_resolution, [1], f"{indir}dm2_rescaled.tif", out_path=root, mesh_ext=ext)
    shutil.move(f"{root}b{base_resolution}_dn1{ext}", f"{meshpath}{ext}")
    shutil.move(f"{root}b{base_resolution}_dn1_st{ext}", f"{meshpath}_st{ext}")
    print(f"- Meshes generated after {round(time.time() - start, 2)} seconds.")

    # compute far topography mesh
    ext = '.vtk'
    fartopo = xr.load_dataset(fartopo_path)
    fartopo -= Rb*1e3
    fartopo.band_data.rio.to_raster(f"{indir}fartopo_rescaled.tif")
    # xr.load_dataset(f"{indir}fartopo_rescaled.tif").band_data.plot(robust=True)
    # plt.show()
    # exit()
    prepare_meshes.make(120, [1], f"{indir}fartopo_rescaled.tif", out_path=root, mesh_ext=ext)
    shutil.move(f"{root}b120_dn1{ext}", f"{fartopomesh}{ext}")
    shutil.move(f"{root}b120_dn1_st{ext}", f"{fartopomesh}_st{ext}")
    print(f"- Far topo meshes generated after {round(time.time() - start, 2)} seconds.")

    # open index
    lnac_index = f"{indir}CUMINDEX_LROC.TAB"
    cumindex = pd.read_csv(lnac_index, index_col=None)

    # get list of images from mapprojected folder
    epos_utc = ['2023-09-29 06:00:00.0']
    print(f"- Rendering input DEM at {epos_utc}.")

    for epo_in in epos_utc:
        dsi, epo_out = render_at_date(meshes={'stereo': f"{meshpath}_st{ext}", 'cart': f"{meshpath}{ext}"},
                       path_to_furnsh=f"{indir}simple.furnsh", epo_utc=epo_in, show=True, basemesh_path=f"{fartopomesh}{ext}")
