import glob
import logging
import os
import shutil
import sys
import time
import pandas as pd
import datetime
import xarray as xr
from tqdm import tqdm
from matplotlib import pyplot as plt

from examples.download_kernels import download_kernels
from src import prepare_meshes
from src.render_dem import render_at_date
from src.flux_util import get_Fsun

if __name__ == '__main__':

    # DM2, S01, Haworth close to ray, De Gerlache S11, Malapert
    siteid = sys.argv[1]

    Rb = 1737.4  # km
    base_resolution = 20
    root = "examples/"
    os.makedirs(root, exist_ok=True)

    # download kernels
    download_kernels()

    # Elevation/DEM GTiff input
    indir = f"{root}aux/"
    outdir = f"{root}out/"
    # tif_path = f'{indir}ldem_6_cut.tif'  #
    #tif_path = f'{indir}{siteid}_final_adj_5mpp_surf.tif'  #
    tif_path = f"/home/sberton2/Lavoro/projects/HabNiches/dems/{siteid}_final_adj_5mpp_surf.tif"
    # flux_path = f"{indir}ssi_v02r01_yearly_s1610_e2022_c20230120.nc" (only covers >115 nm)
    flux_path = f"{indir}ref_solar_irradiance_whi-2008_ver2.dat"
    meshpath = tif_path.split('.')[0]

    # prepare mesh of the input dem
    start = time.time()
    print(f"- Computing trimesh for {tif_path}...")

    # regular delauney mesh
    ext = '.vtk'
    prepare_meshes.make(base_resolution, [1], tif_path, out_path=root, mesh_ext=ext)
    shutil.move(f"{root}b{base_resolution}_dn1{ext}", f"{meshpath}{ext}")
    shutil.move(f"{root}b{base_resolution}_dn1_st{ext}", f"{meshpath}_st{ext}")
    print(f"- Meshes generated after {round(time.time() - start, 2)} seconds.")

    # open index
    lnac_index = f"{indir}CUMINDEX_LROC.TAB"
    cumindex = pd.read_csv(lnac_index, index_col=None)

    # get list of images from mapprojected folder
    # epos_utc = ['2023-09-29 06:00:00.0']
    start_time = datetime.date(2020, 9, 1)
    end_time = datetime.date(2020, 9, 30)
    s = pd.Series(pd.date_range(start_time, end_time, freq='24H')
                  .strftime('%Y-%m-%d %H:%M:%S.%f'))
    epos_utc = s.values.tolist()
    print(f"- Rendering input DEM at {epos_utc}.")

    dsi_list = {}
    for epo_in in tqdm(epos_utc):
        # retrieve UV flux
        Fsun = get_Fsun(flux_path, epo_in, wavelength=[0, 320])
        dsi, epo_out = render_at_date(meshes={'stereo': f"{meshpath}_st{ext}", 'cart': f"{meshpath}{ext}"},
                                      path_to_furnsh=f"{indir}simple.furnsh", epo_utc=epo_in, Fsun=Fsun,
                                      show=True)
        dsi_list[epo_out] = dsi

    list_da = []
    for epo, da in dsi_list.items():
        da = da.assign_coords(time=epo)
        da = da.expand_dims(dim="time")

        list_da.append(da)

    # stack dataarrays in list
    ds = xr.combine_by_coords(list_da)
    moon_sp_crs = xr.open_dataset(tif_path).rio.crs
    ds.rio.write_crs(moon_sp_crs, inplace=True)
    print(ds)

    # get cumulative flux (assuming 24H steps for now)
    step = 24. * 86400.
    dssum = (ds * step).sum(dim='time')
    # get max flux
    dsmax = ds.max(dim='time')
    # get average flux
    dsmean = ds.mean(dim='time')

    # save to raster
    format_code = '%Y%m%d%H%M%S'
    start_time = start_time.strftime(format_code)
    end_time = end_time.strftime(format_code)

    sumout = f"{outdir}{siteid}_sum_{start_time}_{end_time}.tif"
    dssum.flux.rio.to_raster(sumout)
    logging.info(f"- Cumulative flux over {list(dsi_list.keys())[0]} to {list(dsi_list.keys())[-1]} saved to {sumout}.")

    maxout = f"{outdir}{siteid}_max_{start_time}_{end_time}.tif"
    dsmax.flux.rio.to_raster(maxout)
    logging.info(f"- Maximum flux over {list(dsi_list.keys())[0]} to {list(dsi_list.keys())[-1]} saved to {maxout}.")

    meanout = f"{outdir}{siteid}_mean_{start_time}_{end_time}.tif"
    dsmean.flux.rio.to_raster(meanout)
    logging.info(f"- Average flux over {list(dsi_list.keys())[0]} to {list(dsi_list.keys())[-1]} saved to {meanout}.")

    # plot statistics
    fig, axes = plt.subplots(1, 3, figsize=(26, 6))
    dssum.flux.plot(ax=axes[0], robust=True)
    axes[0].set_title('Sum')
    dsmax.flux.plot(ax=axes[1], robust=True)
    axes[1].set_title('Max')
    dsmean.flux.plot(ax=axes[2], robust=True)
    axes[2].set_title('Mean')
    plt.suptitle(f'Statistics of solar flux between {start_time} and {end_time}.')
    plt.show()

