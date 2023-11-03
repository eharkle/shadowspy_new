import glob
import os
import shutil
import time
import pandas as pd
import datetime
import xarray as xr
from tqdm import tqdm
from matplotlib import pyplot as plt

from examples.download_kernels import download_kernels
from src import prepare_meshes
from src.render_dem import render_at_date

if __name__ == '__main__':

    # DM2, S01, Haworth close to ray, De Gerlache S11, Malapert

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
    tif_path = f'{indir}ldem_6_cut.tif'  #
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
    start_time = datetime.date(2023, 9, 1)
    end_time = datetime.date(2023, 9, 30)
    s = pd.Series(pd.date_range(start_time, end_time, freq='6H')
                  .strftime('%Y-%m-%d %H:%M:%S.%f'))
    epos_utc = s.values.tolist()
    print(f"- Rendering input DEM at {epos_utc}.")

    dsi_list = {}
    for epo_in in tqdm(epos_utc):
        dsi, epo_out = render_at_date(meshes={'stereo': f"{meshpath}_st{ext}", 'cart': f"{meshpath}{ext}"},
                                      path_to_furnsh=f"{indir}simple.furnsh", epo_utc=epo_in)
        dsi_list[epo_out] = dsi

    list_da = []
    for epo, da in dsi_list.items():
        da = da.assign_coords(time=epo)
        da = da.expand_dims(dim="time")

        list_da.append(da)

    # stack dataarrays in list
    ds = xr.combine_by_coords(list_da)
    print(ds)

    fig, axes = plt.subplots(1, 3, figsize=(26, 6))
    # get cumulative flux
    ds.sum(dim='time').flux.plot(ax=axes[0])
    axes[0].set_title('Sum')

    # get max flux
    ds.max(dim='time').flux.plot(ax=axes[1])
    axes[1].set_title('Max')

    # get average flux
    ds.mean(dim='time').flux.plot(ax=axes[2])
    axes[2].set_title('Mean')

    plt.suptitle(f'Statistics of solar flux between {start_time} and {end_time}.')
    plt.show()
