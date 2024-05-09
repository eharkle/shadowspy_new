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
from src.shadowspy.raster_products import basic_raster_stats
from src.shadowspy.render_dem import irradiance_at_date
from src.shadowspy.utilities import run_log

def main_pipeline():

    start = time.time()

    def get_config(opt):
        settings = {
            'siteid': opt.siteid,
            'Rb': opt.Rb,
            'base_resolution': opt.base_resolution,
            'max_extension': float(opt.max_extension),
            'extres': {float(k): int(v) for k, v in opt.extres.items()},
            'root': opt.root,
            'indir': opt.indir,
            'dem_path': opt.dem_path,
            'fartopo_path': opt.fartopo_path if opt.fartopo_path not in ['None', 'same_dem']
            else opt.dem_path if opt.fartopo_path == 'same_dem'
            else None,
            'outdir': opt.outdir,
            'tmpdir': opt.tmpdir,
        }
        return settings
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

        if not opt.point_source:
            logging.error("* Can only provide azimuth&elevation when using a point source.")
            exit()

        data_list = pd.read_csv(opt.azi_ele_path).values.tolist()
        print(data_list)

    elif len(opt.epos_utc) > 0:
        data_list = opt.epos_utc

    else:
        use_azi_ele = False
        start_time = datetime.datetime.strptime(opt.start_time, '%Y-%m-%d %H:%M:%S.%f')
        end_time = datetime.datetime.strptime(opt.end_time, '%Y-%m-%d %H:%M:%S.%f')
        s = pd.Series(pd.date_range(start_time, end_time, freq=f'{opt.time_step_hours}H')
                      .strftime('%Y-%m-%d %H:%M:%S.%f'))
        data_list = s.values.tolist()
    print(f"- Illuminating input DEM at {data_list}.")

    # actually compute irradiance at each element of data_list
    dsi_epo_path_dict = {}
    for idx, data in tqdm(enumerate(data_list), total=len(data_list)):

        if use_azi_ele:
            # For azimuth-elevation inputs
            func_args = {'azi_ele_deg': data}
            epo_in = '2000-01-01 00:00:00.0'
        else:
            func_args = {'epo_utc': data}
            epo_in = data

        if opt.flux_path is None:
            Fsun = get_Fsun(opt.flux_path, epo_in, wavelength=opt.wavelength)
        else:
            Fsun = opt.Fsun

        # Common arguments for both cases
        common_args = {
            'meshes': {'stereo': f"{inner_mesh_path}_st{ext}", 'cart': f"{inner_mesh_path}{ext}"},
            'basemesh_path': outer_mesh_path + ext,
            'path_to_furnsh': f"{indir}simple.furnsh",
            'point': opt.point_source,
            'extsource_coord': opt.extsource_coord,
            'source': opt.source,
            'inc_flux': Fsun,
        }

        # Call the function with dynamically constructed arguments
        full_args = {**common_args, **func_args}
        dsi, epo_out = irradiance_at_date(**full_args)

        # get illum epoch string
        epostr = datetime.datetime.strptime(epo_in, '%Y-%m-%d %H:%M:%S.%f')
        epostr = epostr.strftime('%y%m%d%H%M%S')

        # define useful quantities
        outpath = f"{outdir}{siteid}/{siteid}_{epostr}_{idx}"
        dsi_epo_path_dict[epostr] = outpath+'.tif'

        # save each output to raster to save memory
        dsi.rio.write_crs(dem_crs, inplace=True)
        dsi = dsi.assign_coords(time=epo_in)
        dsi = dsi.expand_dims(dim="time")
        dsi = dsi.rio.reproject_match(dem, resampling=Resampling.cubic_spline)
        dsi.flux.rio.to_raster(f"{outpath}.tif", compress='zstd')

    # prepare mean, sum, max stats rasters
    if not use_azi_ele:
        basic_raster_stats(dsi_epo_path_dict, opt.time_step_hours, crs=dem_crs, outdir=outdir, siteid=siteid)

    # set up logs
    run_log(Fsun=Fsun, Rb=Rb, base_resolution=base_resolution, siteid=siteid, dem_path=dem_path, outdir=outdir,
            start_time=opt.start_time, end_time=opt.end_time, time_step_hours=opt.time_step_hours,
            runtime_sec=round(time.time() - start, 2), logpath=f"{outdir}illum_stats_{siteid}_{int(time.time())}.json")

# def main_pipeline(opt, process_function):
#     start = time.time()
#     setup_directories(opt.root, opt.outdir, opt.tmpdir, opt.siteid)
#     meshpath = prepare_dem_mesh(opt.dem_path, opt.tmpdir, opt.siteid, opt)
#
#     use_azi_ele = opt.azi_ele_path not in [None, 'None']
#     data_list = fetch_and_process_data(opt, use_azi_ele)
#
#     common_args = setup_common_args(meshpath, opt)
#     process_function(data_list, common_args, use_azi_ele, opt)
#
#     print(f"Completed in {round(time.time() - start, 2)} seconds.")

if __name__ == '__main__':

    opt = ShSpOpt()
    opt.setup_config()
    main_pipeline(opt, process_data_list)
