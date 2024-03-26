import glob
import os
import shutil
import time

import matplotlib.pyplot as plt
import pandas as pd

import mesh_generation
from examples.download_kernels import download_kernels
from shadowspy.image_util import read_img_properties
from shadowspy.render_dem import render_match_image
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
    fartopo = xr.open_dataset(tif_path)
    fartopo -= Rb*1e3
    fartopo.band_data.rio.to_raster(f"{indir}dm2_rescaled.tif")
    mesh_generation.make(base_resolution, [1], f"{indir}dm2_rescaled.tif", out_path=root, mesh_ext=ext)
    shutil.move(f"{root}b{base_resolution}_dn1{ext}", f"{meshpath}{ext}")
    shutil.move(f"{root}b{base_resolution}_dn1_st{ext}", f"{meshpath}_st{ext}")
    print(f"- Meshes generated after {round(time.time() - start, 2)} seconds.")

    # compute far topography mesh
    ext = '.vtk'
    fartopo = xr.open_dataset(fartopo_path)
    fartopo -= Rb*1e3
    fartopo.band_data.rio.to_raster(f"{indir}fartopo_rescaled.tif")
    # xr.open_dataset(f"{indir}fartopo_rescaled.tif").band_data.plot(robust=True)
    # plt.show()
    # exit()
    mesh_generation.make(120, [1], f"{indir}fartopo_rescaled.tif", out_path=root, mesh_ext=ext)
    shutil.move(f"{root}b120_dn1{ext}", f"{fartopomesh}{ext}")
    shutil.move(f"{root}b120_dn1_st{ext}", f"{fartopomesh}_st{ext}")
    print(f"- Far topo meshes generated after {round(time.time() - start, 2)} seconds.")

    # open index
    lnac_index = f"{indir}CUMINDEX_LROC.TAB"
    cumindex = pd.read_csv(lnac_index, index_col=None)

    # get list of images from mapprojected folder
    imgs = glob.glob(f"{indir}M104299437RE_map.tif")[:]
    imgs_names = [f.split('/')[-1].split('_map')[0] for f in imgs]
    imgs_nam_epo_path = read_img_properties(imgs_names, cumindex)

    imgs_nam_epo_path['meas_path'] = [f"{indir}{img}_map.tif"
                                      for img in imgs_nam_epo_path.PRODUCT_ID.values]
    print(f"- {len(imgs_nam_epo_path)} images found in path. Rendering input DEM.")

    for idx, row in imgs_nam_epo_path.iterrows():
        render_match_image(root, meshes={'stereo': f"{meshpath}_st{ext}", 'cart': f"{meshpath}{ext}"},
                       path_to_furnsh=f"{indir}simple.furnsh",
                       img_name=row[0], epo_utc=row[1], meas_path=row[2],
                       center='P', basemesh=f"{fartopomesh}{ext}")
