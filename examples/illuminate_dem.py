import glob
import os
import shutil
import time
import datetime
import xarray as xr
import pandas as pd
from tqdm import tqdm
import sys

import mesh_generation
from examples.download_kernels import download_kernels
from shadowspy.render_dem import render_at_date
from rasterio.enums import Resampling

if __name__ == '__main__':

    siteid = sys.argv[1] # 'Site23' # 'DM2'

    # compute direct flux from the Sun
    Fsun = 1361  # W/m2
    Rb = 1737.4 # km
    lonlat0_stereo = (0, -90)
    base_resolution = 1
    root = "examples/"
    os.makedirs(root, exist_ok=True)

    # download kernels
    download_kernels()

    # Elevation/DEM GTiff input
    indir = f"{root}aux/"
    tif_path = f'{indir}{siteid}_GLDELEV_001.tif' # _final_adj_5mpp_surf.tif'  #
    meshpath = tif_path.split('.')[0]
    outdir = f"{root}out/"
    os.makedirs(f"{outdir}{siteid}", exist_ok=True)

    # prepare mesh of the input dem
    start = time.time()
    print(f"- Computing trimesh for {tif_path}...")

    # extract crs
    dem = xr.open_dataset(tif_path)
    dem_crs = dem.rio.crs

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
    # epos_utc = ['2023-07-09 17:15:00.0', '2023-07-09 15:17:00.0']
    start_time = datetime.date(2025, 6, 21)
    end_time = datetime.date(2025, 9, 21)
    time_step_hours = 24
    s = pd.Series(pd.date_range(start_time, end_time, freq=f'{time_step_hours}H')
                  .strftime('%Y-%m-%d %H:%M:%S.%f'))
    epos_utc = s.values.tolist()
    print(f"- Rendering input DEM at {epos_utc}.")

    for epo_in in tqdm(epos_utc, desc='rendering each epos_utc', total=len(epos_utc)):
        dsi, epo_out = render_at_date(meshes={'stereo': f"{meshpath}_st{ext}", 'cart': f"{meshpath}{ext}"},
                                      path_to_furnsh=f"{indir}simple.furnsh", epo_utc=epo_in, show=False,
                                      crs=dem_crs, point=True, source='SUN', inc_flux=Fsun)

        # save each output to raster
        dsi = dsi.assign_coords(time=epo_in)
        dsi = dsi.expand_dims(dim="time")
        epostr = datetime.datetime.strptime(epo_in,'%Y-%m-%d %H:%M:%S.%f')
        epostr = epostr.strftime('%y%m%d%H%M%S')
        dsi = dsi.rio.reproject_match(dem, resampling=Resampling.bilinear)
        dsi.flux.rio.to_raster(f"{outdir}{siteid}/{siteid}_GLDSFLX_001_{epostr}_000.tif")
