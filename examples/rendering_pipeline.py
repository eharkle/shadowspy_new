import logging
import os
import shutil
import time
import pandas as pd
import datetime
import xarray as xr
from shapely.geometry import box
from tqdm import tqdm
from matplotlib import pyplot as plt
from rasterio.enums import Resampling

from examples.download_kernels import download_kernels
from src.config import ShSpOpt
from src.mesh_operations import mesh_generation
from src.mesh_operations.helpers import prepare_inner_outer_mesh
from src.shadowspy.flux_util import get_Fsun
from src.shadowspy.image_util import read_img_properties
from src.shadowspy.raster_products import basic_raster_stats
from src.shadowspy.render_dem import irradiance_at_date, render_match_image
from src.shadowspy.utilities import run_log

def rendering_pipeline():
    # compute direct flux from the Sun

    start = time.time()

    opt = ShSpOpt()
    opt.setup_config()

    # set local vars (backward compatibility)
    siteid = opt.siteid
    Rb = opt.Rb
    base_resolution = opt.base_resolution
    max_extension = float(opt.max_extension)
    extres = {float(k):int(v) for k, v in opt.extres.items()}
    root = opt.root
    indir = opt.indir
    dem_path = opt.dem_path

    fartopo_path = opt.fartopo_path if opt.fartopo_path not in ['None', 'same_dem'] \
        else dem_path if opt.fartopo_path == 'same_dem' \
        else None

    outdir = opt.outdir
    tmpdir = opt.tmpdir
    use_azi_ele = False

    # download kernels
    if opt.download_kernels:
        download_kernels()

    # prepare dirs
    os.makedirs(root, exist_ok=True)
    os.makedirs(f"{outdir}{siteid}/", exist_ok=True)
    os.makedirs(tmpdir, exist_ok=True)

    # prepare mesh of the input dem
    start = time.time()
    print(f"- Computing trimesh for {dem_path}...")

    # extract crs
    dem = xr.open_dataarray(dem_path)
    dem_crs = dem.rio.crs
    print(dem_crs)

    if opt.bbox_roi is not None:
        minx = opt.bbox_roi[0]; miny = opt.bbox_roi[1]; maxx = opt.bbox_roi[2]; maxy = opt.bbox_roi[3]
        str_bbox = f"{minx}_{miny}_{maxx}_{maxy}"
        dem_path = f"{tmpdir}clipped_dem_{siteid}_{str_bbox}.tif"
        clipped = dem.rio.clip([box(minx=minx, miny=miny, maxx=maxx, maxy=maxy)])
        clipped.rio.to_raster(dem_path)
        logging.info(f"Clipped {opt.dem_path} to {str_bbox} and saved to {dem_path}.")

    # regular delauney mesh
    ext = '.vtk'
    meshpath = tmpdir+dem_path.split('/')[-1].split('.')[0]
    mesh_generation.make(base_resolution, [1], dem_path, out_path=f"{tmpdir}{siteid}_",
                         mesh_ext=ext, rescale_fact=1e-3, lonlat0=opt.lonlat0_stereo)
    shutil.move(f"{tmpdir}{siteid}_b{base_resolution}_dn1{ext}", f"{meshpath}{ext}")
    shutil.move(f"{tmpdir}{siteid}_b{base_resolution}_dn1_st{ext}", f"{meshpath}_st{ext}")
    print(f"- Meshes generated after {round(time.time() - start, 2)} seconds.")

    # prepare full mesh (inner + outer)
    if fartopo_path is not None:
        len_inner_faces_path = f'{tmpdir}len_inner_faces.txt'
        if os.path.exists(len_inner_faces_path):
            last_ext = max({ext: res for ext, res in extres.items() if ext < max_extension}.keys())
            len_inner_faces = pd.read_csv(len_inner_faces_path, header=None).values[0][0]
            inner_mesh_path = meshpath
            outer_mesh_path = f"{tmpdir}LDEM_{int(last_ext)}M_outer"
        else:
            len_inner_faces, inner_mesh_path, outer_mesh_path = prepare_inner_outer_mesh(dem_path, fartopo_path, extres,
                                                                                         max_extension, Rb, tmpdir,
                                                                                         meshpath, ext)
            with open(len_inner_faces_path, 'w') as f:
                f.write('%d' % len_inner_faces)
    else:
        inner_mesh_path = meshpath
        outer_mesh_path = None

    # Determine the mode and prepare data list
    if opt.azi_ele_path not in [None, 'None']:
        use_azi_ele = True
        data_list = pd.read_csv(opt.azi_ele_path).tolist()
    else:
        use_azi_ele = False
        images_index = opt.images_index
        cumindex = pd.read_csv(images_index, index_col=None)

        # get list of images from mapprojected folder
        imgs_nam_epo_path = read_img_properties(images_index.image_name, cumindex)

        data_list['meas_path'] = [f"{indir}{img}_map.tif"
                                          for img in imgs_nam_epo_path.PRODUCT_ID.values]
    print(f"- Rendering input DEM at {data_list}.")

    # open index to get images info

    print(f"- {len(imgs_nam_epo_path)} images found in path. Rendering input DEM.")

    # actually compute irradiance at each element of data_list
    dsi_epo_path_dict = {}
    for idx, data in tqdm(enumerate(data_list), total=len(data_list)):

        if use_azi_ele:
            # For azimuth-elevation inputs
            func_args = {'azi_ele': data}
        else:
            func_args = {'img_name': data[0], 'epo_utc': data[1], 'meas_path': data[2]}

        # Common arguments for both cases
        common_args = {
            'meshes': {'stereo': f"{inner_mesh_path}_st{ext}", 'cart': f"{inner_mesh_path}{ext}"},
            'basemesh_path': outer_mesh_path + ext,
            'path_to_furnsh': f"{indir}simple.furnsh",
            'point': opt.point_source,
            'extsource_coord': opt.extsource_coord,
            'source': opt.source,
            'inc_flux': opt.Fsun,
        }

        # Call the function with dynamically constructed arguments
        full_args = {**common_args, **func_args}
        render_match_image(**full_args)


if __name__ == '__main__':

    rendering_pipeline()