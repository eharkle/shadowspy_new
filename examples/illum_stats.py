import sys
import glob
import json
import logging
import os
import shutil
import time
import pandas as pd
import datetime
import xarray as xr
from tqdm import tqdm
from matplotlib import pyplot as plt

from examples.download_kernels import download_kernels
from shadowspy import prepare_meshes
from shadowspy.render_dem import render_at_date, irradiance_at_date

if __name__ == '__main__':

    start = time.time()

    # DM2, S01, Haworth close to ray, De Gerlache S11, Malapert
    siteid = sys.argv[1] # 'Site23' # 'DM2'

    # compute direct flux from the Sun
    Fsun = 1361  # W/m2
    Rb = 1737.4  # km
    base_resolution = 5
    root = "examples/"
    os.makedirs(root, exist_ok=True)

    # download kernels
    download_kernels()

    # Elevation/DEM GTiff input
    indir = f"{root}aux/"
    tif_path = f'{indir}{siteid}_GLDELEV_001.tif' # _final_adj_5mpp_surf.tif'  #
    outdir = f"{root}out/"
    os.makedirs(f"{outdir}{siteid}/", exist_ok=True)
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
    start_time = datetime.date(2024, 2, 1)
    end_time = datetime.date(2024, 2, 28)
    time_step_hours = 24
    s = pd.Series(pd.date_range(start_time, end_time, freq=f'{time_step_hours}H')
                  .strftime('%Y-%m-%d %H:%M:%S.%f'))
    epos_utc = s.values.tolist()
    print(f"- Rendering input DEM at {epos_utc}.")

    dsi_list = {}
    for idx, epo_in in tqdm(enumerate(epos_utc)):
        dsi, epo_out = irradiance_at_date(meshes={'stereo': f"{meshpath}_st{ext}", 'cart': f"{meshpath}{ext}"},
                                            path_to_furnsh=f"{indir}simple.furnsh", epo_utc=epo_in,
                                          point=True, source='SUN', inc_flux=Fsun)

        # save each output to raster to save memory
        dsi = dsi.assign_coords(time=epo_in)
        dsi = dsi.expand_dims(dim="time")
        dsi.flux.rio.to_raster(f"{outdir}{siteid}/{siteid}_{idx}.tif")
        dsi.flux.plot(clim=[0, 350])
        plt.savefig(f'{outdir}{siteid}/{siteid}_illum_{epo_in}_{idx}.png')
        plt.clf()
        
    # load and stack dataarrays from list
    list_da = []
    for idx, epo in tqdm(enumerate(epos_utc)): #dsi_list.items():
        da = xr.load_dataset(f"{outdir}{siteid}/{siteid}_{idx}.tif")
        da = da.assign_coords(time=epo)
        da = da.expand_dims(dim="time")
        da['flux'] = da.band_data
        da = da.drop_vars("band_data")
        list_da.append(da)

    # stack dataarrays in list
    ds = xr.combine_by_coords(list_da)
    moon_sp_crs = xr.open_dataset(tif_path).rio.crs
    ds.rio.write_crs(moon_sp_crs, inplace=True)
    print(ds)

    # get cumulative flux
    step_sec = time_step_hours * 3600.
    dssum = (ds * step_sec).sum(dim='time')
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
    logging.info(f"- Cumulative flux saved to {sumout}.")

    maxout = f"{outdir}{siteid}_max_{start_time}_{end_time}.tif"
    dsmax.flux.rio.to_raster(maxout)
    logging.info(f"- Maximum flux saved to {maxout}.")

    meanout = f"{outdir}{siteid}_mean_{start_time}_{end_time}.tif"
    dsmean.flux.rio.to_raster(meanout)
    logging.info(f"- Average flux over saved to {meanout}.")

    # plot statistics
    fig, axes = plt.subplots(1, 3, figsize=(26, 6))
    dssum.flux.plot(ax=axes[0], robust=True)
    axes[0].set_title(r'Sum (J/m$^2$)')
    dsmax.flux.plot(ax=axes[1], robust=True)
    axes[1].set_title(r'Max (W/m$^2$)')
    dsmean.flux.plot(ax=axes[2], robust=True)
    axes[2].set_title(r'Mean (W/m$^2$)')
    plt.suptitle(f'Statistics of solar flux at {siteid} between {start_time} and {end_time}.')
    pngout = f"{outdir}{siteid}_stats_{start_time}_{end_time}.png"
    plt.savefig(pngout)
    # plt.show()

    # set up logs
    log_dict = {}
    log_dict['Fsun'] = Fsun
    log_dict['Rb'] = Rb
    log_dict['base_resolution'] = base_resolution
    log_dict['tif_path'] = f'{indir}{siteid}_final_adj_5mpp_surf.tif'
    log_dict['outdir'] = f"{root}out/"
    log_dict['start_time'] = start_time
    log_dict['end_time'] = end_time
    log_dict['time_step_hours'] = time_step_hours
    log_dict['runtime_sec'] = round(time.time() - start, 2)
    with open(f"{outdir}illum_stats_{siteid}_{int(time.time())}", "w") as fp:
        json.dump(log_dict, fp, indent=4)


