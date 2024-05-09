import os
import shutil
import time
import pandas as pd
import datetime
import xarray as xr
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
from src.shadowspy.utils import run_log

if __name__ == '__main__':
    # compute direct flux from the Sun

    start = time.time()

    opt = ShSpOpt()
    opt.setup_config()

    # set local vars (backward compatibility)
    siteid = opt.siteid
    Rb = opt.Rb
    base_resolution = opt.base_resolution
    max_extension = opt.max_extension
    extres = opt.extres
    root = opt.root
    indir = opt.indir
    dem_path = opt.dem_path
    if opt.fartopo_path is not None:
        fartopo_path = opt.fartopo_path
    else:
        fartopo_path = None # dem_path
    outdir = opt.outdir
    tmpdir = opt.tmpdir

    # download kernels
    if opt.download_kernels:
        download_kernels()

    # Elevation/DEM GTiff input
    meshpath = dem_path.split('.')[0]

    # prepare dirs
    os.makedirs(root, exist_ok=True)
    os.makedirs(f"{outdir}{siteid}/", exist_ok=True)
    os.makedirs(tmpdir, exist_ok=True)

    # prepare mesh of the input dem
    start = time.time()
    print(f"- Computing trimesh for {dem_path}...")

    # extract crs
    dem = xr.open_dataset(dem_path)
    dem_crs = dem.rio.crs
    
    # regular delauney mesh
    ext = '.vtk'
    mesh_generation.make(base_resolution, [1], dem_path, out_path=f"{indir}{siteid}_",
                         mesh_ext=ext, rescale_fact=1e-3, lonlat0=opt.lonlat0_stereo)
    shutil.move(f"{root}b{base_resolution}_dn1{ext}", f"{meshpath}{ext}")
    shutil.move(f"{root}b{base_resolution}_dn1_st{ext}", f"{meshpath}_st{ext}")
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

    # get list of images from mapprojected folder
    if opt.epos_utc is None:
        s = pd.Series(pd.date_range(opt.start_time, opt.end_time, freq=f'{opt.time_step_hours}H')
                      .strftime('%Y-%m-%d %H:%M:%S.%f'))
        epos_utc = s.values.tolist()
    else:
        epos_utc = opt.epos_utc
    print(f"- Rendering input DEM at {epos_utc}.")

    # actually compute irradiance at each epoch in epos_utc
    dsi_epo_path_dict = {}
    for idx, epo_in in tqdm(enumerate(epos_utc), total=len(epos_utc)):

        if opt.flux_path is None:
            Fsun = get_Fsun(opt.flux_path, epo_in, wavelength=opt.wavelength)
        else:
            Fsun = opt.Fsun

        dsi, epo_out = irradiance_at_date(meshes={'stereo': f"{meshpath}_st{ext}", 'cart': f"{meshpath}{ext}"},
                                          path_to_furnsh=f"{indir}simple.furnsh", epo_utc=epo_in,
                                          point=True, source='SUN', inc_flux=Fsun)

        # get illum epoch string
        epostr = datetime.strptime(epo_in,'%Y-%m-%d %H:%M:%S.%f')
        epostr = epostr.strftime('%y%m%d%H%M%S')

        # define useful quantities
        outpath = f"{outdir}{siteid}/{siteid}_{epostr}_{idx}"
        dsi_epo_path_dict[epostr] = outpath

        # save each output to raster to save memory
        dsi = dsi.assign_coords(time=epo_in)
        dsi = dsi.expand_dims(dim="time")
        dsi = dsi.rio.reproject_match(dem, resampling=Resampling.bilinear)
        dsi.flux.rio.to_raster(f"{outpath}.tif")
        dsi.flux.plot(clim=[0, 350])
        plt.savefig(f"{outpath}.png")
        plt.clf()

    # prepare mean, sum, max stats rasters
    basic_raster_stats(dsi_epo_path_dict, opt.time_step_hours, crs=dem_crs, outdir=outdir, siteid=siteid)

    # set up logs
    run_log(Fsun=Fsun, Rb=Rb, base_resolution=base_resolution, siteid=siteid, dem_path=dem_path, outdir=outdir,
            start_time=opt.start_time, end_time=opt.end_time, time_step_hours=opt.time_step_hours,
            runtime_sec=round(time.time() - start, 2), logpath=f"{outdir}illum_stats_{siteid}_{int(time.time())}.json")


