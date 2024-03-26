import glob
import os
import shutil
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pandas as pd
from tqdm import tqdm
from rasterio.enums import Resampling

import mesh_generation
from examples.download_kernels import download_kernels
from shadowspy.render_dem import render_at_date

if __name__ == '__main__':

    # compute direct flux from the Sun
    Fsun = 1361  # W/m2
    Rb = 1737.4 # km
    lonlat0_stereo = (0, 90)
    base_resolution = 60
    root = "examples/"
    os.makedirs(root, exist_ok=True)

    # download kernels
    download_kernels()

    # Elevation/DEM GTiff input
    indir = f"{root}aux/"
    tif_path = f'{indir}np0_20mpp_small.tif'#
    meshpath = tif_path.split('.')[0]
    outdir = f"{root}out/"
    siteid = 'np0'
    os.makedirs(f"{outdir}{siteid}", exist_ok=True)

    # prepare mesh of the input dem
    start = time.time()
    print(f"- Computing trimesh for {tif_path}...")

    # extract crs
    dem = xr.open_dataset(tif_path)
    dem_crs = dem.rio.crs

    small_tif = xr.open_dataset(tif_path).isel(band=0, x=slice(None, 200), y=slice(None, 200))
    print(small_tif)
    print(small_tif.rio.resolution())
    tif_path = f'{indir}np0_20mpp_smaller.tif'#
    small_tif.rio.to_raster(tif_path)

    # regular delauney mesh
    ext = '.vtk'
    mesh_generation.make(base_resolution, [1], tif_path, out_path=root, mesh_ext=ext,
                         plarad=Rb, lonlat0=lonlat0_stereo)
    shutil.move(f"{root}b{base_resolution}_dn1{ext}", f"{meshpath}{ext}")
    shutil.move(f"{root}b{base_resolution}_dn1_st{ext}", f"{meshpath}_st{ext}")
    print(f"- Meshes generated after {round(time.time() - start, 2)} seconds.")

    # # open index
    # lnac_index = f"{indir}CUMINDEX_LROC.TAB"
    # cumindex = pd.read_csv(lnac_index, index_col=None)

    # get list of images from mapprojected folder
    azi_ele_list = [(0, 30), (0, 90), (45, 30)] # (50.67717174965302, 3.3533658872987755)]
    epo_in = '2000-01-01 00:00:00.0'
    print(f"- Rendering input DEM for (azi, ele)={azi_ele_list}.")

    for azi_ele in tqdm(azi_ele_list, desc='rendering each epos_utc'):
        dsi, epo_out = render_at_date(meshes={'stereo': f"{meshpath}_st{ext}", 'cart': f"{meshpath}{ext}"},
                                      epo_utc=epo_in, path_to_furnsh=f"{indir}simple.furnsh", crs=dem_crs, show=False,
                                      azi_ele_deg=azi_ele)

        # save each output to raster
        dsi = dsi.assign_coords(time=epo_in)
        dsi = dsi.expand_dims(dim="time")
        epostr = datetime.strptime(epo_in,'%Y-%m-%d %H:%M:%S.%f')
        epostr = epostr.strftime('%d%m%Y%H%M%S')
        dsi = dsi.rio.reproject_match(dem, resampling=Resampling.bilinear)
        #dsi.flux.plot(robust=True)
        #plt.show()
        dsi.flux.rio.to_raster(f"{outdir}{siteid}/{siteid}_{azi_ele[0]}_{azi_ele[1]}.tif")
