import os
import shutil
import time
from datetime import datetime, date

import matplotlib.pyplot as plt
import pandas as pd

from rasterio._io import Resampling
from tqdm import tqdm
import xarray as xr
import rioxarray

from examples.download_kernels import download_kernels
from helpers import prepare_inner_outer_mesh
from mesh_operations import mesh_generation
from shadowspy.render_dem import irradiance_at_date

def main():

    # compute direct flux from the Sun
    Fsun = 1361  # W/m2
    Rb = 1737.4 # km
    base_resolution = 10
    max_extension = 400e3
    extres = {20e3: 120, 60e3: 60, 100e3: 120, 150e3: 240, 300e3: 480}
    extres = {20e3: 60, 60e3: 120, 100e3: 240, 300e3: 480}
    root = "examples/"
    os.makedirs(root, exist_ok=True)

    # download kernels
    download_kernels()

    # Elevation/DEM GTiff input
    indir = f"{root}aux/"
    experiment = 'IM1'
    outdir = f"{root}outR/"
    os.makedirs(outdir, exist_ok=True)
    tmpdir = f"{root}tmp/"
    os.makedirs(tmpdir, exist_ok=True)

    tif_path = "/home/sberton2/Scaricati/ldem_0.tif" # f"{indir}IM1_Terry.tif"
    meshpath = tif_path.split('.')[0]
    fartopo_path = "/home/sberton2/Scaricati/LDEM_60000.0KM_outer.tif"
    # fartopo_path = f"{indir}LDEM_80S_80MPP_ADJ.TIF" # f"{indir}IM1_ldem_large.tif"
    # fartopo_path = "/explore/nobackup/people/mkbarker/GCD/grid/20mpp/v4/public/final/LDEM_80S_20MPP_ADJ.TIF"
    # fartopomesh = fartopo_path.split('.')[0]
    ext = '.vtk'

    # prepare mesh of the input dem
    start = time.time()
    print(f"- Computing trimesh for {tif_path}...")

    # Generate uniform meshes for inner...
    mesh_generation.make(base_resolution, [1], tif_path, out_path=tmpdir, mesh_ext=ext,
                         rescale_fact=1e-3, lonlat0=(0, -90))
    shutil.move(f"{tmpdir}b{base_resolution}_dn1{ext}", f"{meshpath}{ext}")
    shutil.move(f"{tmpdir}b{base_resolution}_dn1_st{ext}", f"{meshpath}_st{ext}")

    # prepare full mesh (inner + outer)
    len_inner_faces_path = f'{tmpdir}len_inner_faces.txt'
    if os.path.exists(len_inner_faces_path):
        last_ext = max({ext: res for ext, res in extres.items() if ext < max_extension}.keys())
        len_inner_faces = pd.read_csv(len_inner_faces_path, header=None).values[0][0]
        inner_mesh_path = meshpath
        outer_mesh_path = f"{tmpdir}LDEM_{int(last_ext)}M_outer"
    else:
        len_inner_faces, inner_mesh_path, outer_mesh_path = prepare_inner_outer_mesh(tif_path, fartopo_path, extres,
                                                                                     max_extension, Rb, tmpdir,
                                                                                     meshpath, ext)
        with open(len_inner_faces_path, 'w') as f:
            f.write('%d' % len_inner_faces)

    # get list of images from mapprojected folder
    # epos_utc = ['2024-02-22 23:24:00.0']
    start_time = datetime(2024, 8, 1, 23, 24, 00)
    end_time = datetime(2024, 8, 29, 23, 24, 00)
    time_step_hours = 24*7
    s = pd.Series(pd.date_range(start_time, end_time, freq=f'{time_step_hours}H')
                  .strftime('%Y-%m-%d %H:%M:%S.%f'))
    epos_utc = s.values.tolist()
    print(f"- Rendering input DEM at {epos_utc} on {len_inner_faces} triangles.")

    dem = xr.open_dataarray(tif_path)
    demcrs = dem.rio.crs

    for epo_in in tqdm(epos_utc, total=len(epos_utc)):

        # plot with inner topo only
        dsi, epo_out = irradiance_at_date(meshes={'stereo': f"{inner_mesh_path}_st{ext}", 'cart': f"{inner_mesh_path}{ext}"},
                                            path_to_furnsh=f"{indir}simple.furnsh", epo_utc=epo_in,
                                          point=True, source='SUN', inc_flux=Fsun)

        # plot with inner+outer topo
        dsi_far, epo_out = irradiance_at_date(meshes={'stereo': f"{inner_mesh_path}_st{ext}", 'cart': f"{inner_mesh_path}{ext}"},
                                          epo_utc=epo_in, path_to_furnsh=f"{indir}simple.furnsh",
                                          basemesh_path=f"{outer_mesh_path}{ext}",
                                          point=True, source='SUN', inc_flux=Fsun)

        epostr = datetime.strptime(epo_in,'%Y-%m-%d %H:%M:%S.%f')
        epostr = epostr.strftime('%y%m%d%H%M%S')

        fig, axes = plt.subplots(1, 2, sharex=False, sharey=False, figsize=(15, 5) )

        # rearrange axes, match dem, and save
        dsi['x'] = dsi['x']#*1e-3
        dsi['y'] = dsi['y']#*1e-3
        dsi.rio.write_crs(demcrs, inplace=True)
        dsi.rio.reproject_match(dem, resampling=Resampling.bilinear, inplace=True)
        dsi.flux.rio.to_raster(f"{meshpath}_dsi.tif")

        # plot
        dsi.flux.plot(robust=True, ax=axes[0])
        axes[0].set_title('New inner mesh only')

        # rearrange axes, match dem, and save
        dsi_far['x'] = dsi_far['x']#*1e-3
        dsi_far['y'] = dsi_far['y']#*1e-3
        dsi_far.rio.write_crs(demcrs, inplace=True)
        target_resolution = int(round(dem.rio.resolution()[0],0))
        dsi_far.rio.reproject_match(dem, resampling=Resampling.bilinear, inplace=True)
        dsi_far = dsi_far.rio.reproject(dsi_far.rio.crs, resolution=target_resolution, resampling=Resampling.bilinear)
        # dsi_far.flux.rio.to_raster(f"{outdir}{experiment}_GLDSFLX_001_{epostr}_000.tif")

        # plot
        dsi_far.flux.plot(robust=True, ax=axes[1])
        axes[1].set_title('New inner mesh + ldem merged')

        plt.savefig(f"{outdir}{experiment}_GLDSFLX_001_{epostr}_000.png")
        # plt.show()
        print(f"done {epo_in}")

if __name__ == '__main__':
    main()