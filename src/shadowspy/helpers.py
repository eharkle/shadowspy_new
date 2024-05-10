import logging
import os
import datetime
import xarray as xr
import rioxarray
import pandas as pd
from rasterio._io import Resampling
from tqdm import tqdm

from src.shadowspy.flux_util import get_Fsun
from src.shadowspy.image_util import read_img_properties
from src.shadowspy.render_dem import irradiance_at_date


def setup_directories(opt):
    # prepare dirs
    os.makedirs(opt.root, exist_ok=True)
    os.makedirs(f"{opt.outdir}{opt.siteid}/", exist_ok=True)
    os.makedirs(opt.tmpdir, exist_ok=True)


def process_data_list(data_list, common_args, use_azi_ele, opt):
    dsi_epo_path_dict = {}
    dem = xr.open_dataarray(common_args['dem_path'])

    for data in tqdm(data_list, total=len(data_list)):
        common_args, func_args = prepare_processing(use_azi_ele, data, common_args, opt)
        full_args = {**common_args, **func_args}
        dsi, date_illum_str = irradiance_at_date(**full_args)  # Assume a modified version handling both cases
        dump_processing_results(dsi, dsi_epo_path_dict, dem, func_args, opt)

    return dsi_epo_path_dict


def prepare_processing(use_azi_ele, data, common_args, opt):
    if use_azi_ele:
        # For azimuth-elevation inputs
        func_args = {'azi_ele_deg': data, 'epo_in': '2000-01-01 00:00:00.0'}
    elif opt.images_index not in [None, 'None']:
        images_index = opt.images_index
        cumindex = pd.read_csv(images_index, index_col=None)
        # get list of images from cumindex
        imgs_nam_epo_path = read_img_properties(images_index.image_name, cumindex)
        data_list['meas_path'] = [f"{opt.indir}{img}_map.tif"
                                          for img in imgs_nam_epo_path.PRODUCT_ID.values]
        func_args = {'epo_utc': data, 'epo_in': data}
    else:
        func_args = {'epo_utc': data, 'epo_in': data}

    if opt.flux_path not in [None, 'None']:
        Fsun = get_Fsun(opt.flux_path, func_args['epo_in'], wavelength=opt.wavelength)
    else:
        Fsun = opt.Fsun

    common_args['inc_flux'] = Fsun

    return common_args, func_args


def dump_processing_results(dsi, dsi_epo_path_dict, dem, func_args, opt):
    # get illum epoch string
    try:
        epostr = f"{func_args['azi_ele_deg'][0]}_{func_args['azi_ele_deg'][1]}"
    except:
        epostr = datetime.datetime.strptime(func_args['epo_in'], '%Y-%m-%d %H:%M:%S.%f')
        epostr = epostr.strftime('%y%m%d%H%M%S')

    # define useful quantities
    outpath = f"{opt.outdir}{opt.siteid}/{opt.siteid}_{epostr}"
    dsi_epo_path_dict[epostr] = outpath + '.tif'

    # save each output to raster to save memory
    dsi.rio.write_crs(dem.rio.crs, inplace=True)
    dsi = dsi.assign_coords(time=func_args['epo_in'])
    dsi = dsi.expand_dims(dim="time")
    dsi = dsi.rio.reproject_match(dem, resampling=Resampling.cubic_spline)
    dsi.flux.rio.to_raster(f"{outpath}.tif", compress='zstd')