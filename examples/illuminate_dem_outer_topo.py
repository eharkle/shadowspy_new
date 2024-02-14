import os
import shutil
import time

import matplotlib.pyplot as plt
from rasterio._io import Resampling
from tqdm import tqdm
import xarray as xr
import rioxarray

import mesh_generation
from examples.download_kernels import download_kernels
from mesh_operations import load_mesh
from mesh_operations.merge_overlapping import merge_inout
from shadowspy.render_dem import render_at_date
from split_merged import split_merged


if __name__ == '__main__':

    # compute direct flux from the Sun
    Fsun = 1361  # W/m2
    Rb = 1737.4 # km
    base_resolution = 10
    outer_mesh_resolution = 40
    root = "examples/"
    os.makedirs(root, exist_ok=True)

    # download kernels
    download_kernels()

    # Elevation/DEM GTiff input
    indir = f"{root}aux/"
    tif_path = f"{indir}IM05_GLDELEV_001.tif"
    meshpath = tif_path.split('.')[0]
    fartopo_path = f"{indir}IM1_ldem_large.tif"
    fartopomesh = fartopo_path.split('.')[0]
    ext = '.vtk'

    # prepare mesh of the input dem
    start = time.time()
    print(f"- Computing trimesh for {tif_path}...")

    # Generate uniform meshes for inner...
    mesh_generation.make(base_resolution, [1], tif_path, out_path=root, mesh_ext=ext,
                         rescale_fact=1)
    shutil.move(f"{root}b{base_resolution}_dn1{ext}", f"{meshpath}{ext}")
    shutil.move(f"{root}b{base_resolution}_dn1_st{ext}", f"{meshpath}_st{ext}")
    # ... and outer topography
    mesh_generation.make(outer_mesh_resolution, [1], fartopo_path, out_path=root, mesh_ext=ext,
                         rescale_fact=1)
    shutil.move(f"{root}b{outer_mesh_resolution}_dn1{ext}", f"{fartopomesh}{ext}")
    shutil.move(f"{root}b{outer_mesh_resolution}_dn1_st{ext}", f"{fartopomesh}_st{ext}")
    print(f"- Meshes generated after {round(time.time() - start, 2)} seconds.")

    start = time.time()
    # Merge inner and outer meshes seamlessly
    stacked_mesh_path = f"{indir}stacked_st{ext}"
    input_totalmesh, labels_dict = merge_inout(load_mesh(f"{meshpath}_st{ext}"),
                                               load_mesh(f"{fartopomesh}_st{ext}"),
                                               output_path=stacked_mesh_path)
    print(f"- Meshes merged after {round(time.time() - start, 2)} seconds.")

    start = time.time()
    # Split inner and outer meshes
    len_inner_faces = labels_dict['inner']
    inner_mesh_path, outer_mesh_path = split_merged(input_totalmesh, len_inner_faces, meshpath, fartopomesh, ext, Rb)
    print(f"- Inner+outer meshes generated from merged after {round(time.time() - start, 2)} seconds.")
    #####

    # get list of images from mapprojected folder
    epos_utc = ['2023-09-15 06:00:00.0']
    print(f"- Rendering input DEM at {epos_utc}.")

    dem = xr.load_dataarray(tif_path)
    demcrs = dem.rio.crs

    for epo_in in tqdm(epos_utc, total=len(epos_utc)):

        # plot with inner topo only
        dsi, epo_out = render_at_date(meshes={'stereo': f"{inner_mesh_path}_st{ext}", 'cart': f"{inner_mesh_path}{ext}"},
                                      epo_utc=epo_in, path_to_furnsh=f"{indir}simple.furnsh",
                                      show=False, point=True)

        # plot with inner+outer topo
        dsi_far, epo_out = render_at_date(meshes={'stereo': f"{inner_mesh_path}_st{ext}", 'cart': f"{inner_mesh_path}{ext}"},
                                      epo_utc=epo_in, path_to_furnsh=f"{indir}simple.furnsh",
                                      basemesh_path=f"{outer_mesh_path}{ext}",
                                      show=False, point=True)

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
        dsi_far.flux.rio.to_raster(f"{meshpath}_dsifar.tif")

        # plot
        dsi_far.flux.plot(robust=True, ax=axes[1])
        axes[1].set_title('New inner mesh + ldem merged')

        plt.savefig(f"{indir}renderings.png")
        plt.show()
