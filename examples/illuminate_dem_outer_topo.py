import os
import shutil
import time
from datetime import datetime

import matplotlib.pyplot as plt
from rasterio._io import Resampling
from tqdm import tqdm
import xarray as xr
import rioxarray

from examples.download_kernels import download_kernels
from mesh_operations import load_mesh, mesh_generation
from mesh_operations.merge_overlapping import merge_inout
from shadowspy.render_dem import render_at_date, irradiance_at_date
from mesh_operations.split_merged import split_merged

if __name__ == '__main__':

    # compute direct flux from the Sun
    Fsun = 1361  # W/m2
    Rb = 1737.4 # km
    base_resolution = 10
    max_extension = 50e3
    root = "examples/"
    os.makedirs(root, exist_ok=True)

    # download kernels
    download_kernels()

    # Elevation/DEM GTiff input
    indir = f"{root}aux/"
    tif_path = f"{indir}IM1_Terry.tif"
    meshpath = tif_path.split('.')[0]
    fartopo_path = f"{indir}LDEM_80S_80MPP_ADJ.TIF" # f"{indir}IM1_ldem_large.tif"
    # fartopo_path = "/explore/nobackup/people/mkbarker/GCD/grid/20mpp/v4/public/final/LDEM_80S_20MPP_ADJ.TIF"
    fartopomesh = fartopo_path.split('.')[0]
    ext = '.vtk'

    experiment = 'IM1'
    outdir = f"{root}out/"
    os.makedirs(outdir, exist_ok=True)

    # crop fartopo to box around dem to render
    import numpy as np
    da = xr.load_dataarray(tif_path)
    print(da)
    bounds = da.rio.bounds()
    demcx, demcy = np.mean([bounds[0], bounds[2]]), np.mean([bounds[1], bounds[3]])

    # da = da.rio.clip_box(minx=demcx-300e3, miny=demcy-300e3,
    #                      maxx=demcx+300e3, maxy=demcy+300e3)
    # fartopo_path = f"{indir}LDEM_50KM_80M.tif"
    # fartopomesh = fartopo_path.split('.')[0]
    # da.rio.to_raster(fartopo_path)

    # prepare mesh of the input dem
    start = time.time()
    print(f"- Computing trimesh for {tif_path}...")

    # Generate uniform meshes for inner...
    mesh_generation.make(base_resolution, [1], tif_path, out_path=root, mesh_ext=ext,
                         rescale_fact=1e-3, lonlat0=(0, -90))
    shutil.move(f"{root}b{base_resolution}_dn1{ext}", f"{meshpath}{ext}")
    shutil.move(f"{root}b{base_resolution}_dn1_st{ext}", f"{meshpath}_st{ext}")

    start = time.time()
    da_out = xr.load_dataarray(fartopo_path)
    print(da_out)
    min_resolution = int(round(da_out.rio.resolution()[0],0))
    # Merge inner and outer meshes seamlessly
    # set a couple of layers at 1, 5 and max_extension km ranges
    outer_topos = []
    extres = {20e3: 20, 60e3: 60, 100e3: 120, 150e3: 240, 300e3: 480}
    extres = {ext: max(res, min_resolution) for ext, res in extres.items()
              if ext < max_extension}
    for extension, resol in extres.items():
        da_red = da_out.rio.clip_box(minx=demcx-extension, miny=demcy-extension,
                             maxx=demcx+extension, maxy=demcy+extension)
        fartopo_path = f"{indir}LDEM_{extension}KM_outer.tif"
        fartopomesh = fartopo_path.split('.')[0]
        da_red.rio.to_raster(fartopo_path)
        outer_topos.append({resol: f"{indir}LDEM_{extension}KM_outer.tif"})

    # for iter 0, set inner mesh as stacked mesh
    shutil.copy(f"{meshpath}_st{ext}", f"{indir}stacked_st{ext}")
    labels_dict_list = {}
    for idx, resol_dempath in enumerate(outer_topos):
        resol = list(resol_dempath.keys())[0]
        dempath = list(resol_dempath.values())[0]

        outer_mesh_resolution = resol
        fartopo_path = dempath
        fartopomesh = fartopo_path.split('.')[0]

        print(f"- Adding {fartopo_path} ({fartopomesh}) at {outer_mesh_resolution}mpp.")

        # ... and outer topography
        mesh_generation.make(outer_mesh_resolution, [1], fartopo_path, out_path=root, mesh_ext=ext,
                             rescale_fact=1e-3, lonlat0=(0, -90))
        shutil.move(f"{root}b{outer_mesh_resolution}_dn1_st{ext}", f"{fartopomesh}_st{ext}")
        print(f"- Meshes generated after {round(time.time() - start, 2)} seconds.")

        stacked_mesh_path = f"{indir}stacked_st{ext}"
        input_totalmesh, labels_dict = merge_inout(load_mesh(stacked_mesh_path),
                                                   load_mesh(f"{fartopomesh}_st{ext}"),
                                                   output_path=stacked_mesh_path) #, debug=True)
        labels_dict_list[idx] = labels_dict
        print(f"- Meshes merged after {round(time.time() - start, 2)} seconds and saved to {stacked_mesh_path}.")

    start = time.time()
    # Split inner and outer meshes
    len_inner_faces = labels_dict_list[0]['inner']
    inner_mesh_path, outer_mesh_path = split_merged(input_totalmesh, len_inner_faces, meshpath, fartopomesh, ext, Rb)
    print(f"- Inner+outer meshes generated from merged after {round(time.time() - start, 2)} seconds.")
    #####

    # get list of images from mapprojected folder
    epos_utc = ['2024-02-22 23:24:00.0'] # '2023-09-15 06:00:00.0']
    print(f"- Rendering input DEM at {epos_utc} on {len_inner_faces} triangles.")

    dem = xr.load_dataarray(tif_path)
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
        dsi['x'] = dsi['x']*1e-3
        dsi['y'] = dsi['y']*1e-3
        dsi.rio.write_crs(demcrs, inplace=True)
        dsi.rio.reproject_match(dem, resampling=Resampling.bilinear, inplace=True)
        dsi.flux.rio.to_raster(f"{meshpath}_dsi.tif")

        # plot
        dsi.flux.plot(robust=True, ax=axes[0])
        axes[0].set_title('New inner mesh only')

        # rearrange axes, match dem, and save
        dsi_far['x'] = dsi_far['x']*1e-3
        dsi_far['y'] = dsi_far['y']*1e-3
        dsi_far.rio.write_crs(demcrs, inplace=True)
        dsi_far.rio.reproject_match(dem, resampling=Resampling.bilinear, inplace=True)
        dsi_far.flux.rio.to_raster(f"{outdir}{experiment}_GLDSFLX_001_{epostr}_000.tif")

        # plot
        dsi_far.flux.plot(robust=True, ax=axes[1])
        axes[1].set_title('New inner mesh + ldem merged')

        plt.savefig(f"{outdir}{experiment}_GLDSFLX_001_{epostr}_000.png")
        plt.show()
