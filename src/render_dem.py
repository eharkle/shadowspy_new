import os
from datetime import datetime
import numpy as np
import pandas as pd

from src.import_mesh import import_mesh
from src.spice_util import get_sunvec
from src.shape import CgalTrimeshShapeModel #, EmbreeTrimeshShapeModel
import xarray as xr
from rasterio.enums import Resampling

from src.photometry import mmpf_mh_boyd2017lpsc
from src.math_util import angle_btw


def plot3d(mesh_path, var_to_plot, center='P'):

    import pyvista as pv
    if center == 'P':
        grid = pv.read(f"{mesh_path}")
        grid.cell_data[''] = np.nan
        grid.cell_data[''][:] = 0
        grid.cell_data[''][:] = var_to_plot
        grid.plot(show_scalar_bar=True, show_axes=True, cpos='xy')
    elif center == 'V':
        mesh = pv.read(f"{mesh_path}")
        pl = pv.Plotter()
        pl.add_mesh(mesh, show_edges=False)
        surf_points = mesh.extract_surface().points
        pl.add_points(surf_points, scalars=var_to_plot,
                      point_size=10)
        pl.show(cpos='xy')


def extended_sun(sun_vecs, extsun_coord):
    import csv

    extsun_ = []
    with open(extsun_coord) as csvDataFile:
        csvReader = csv.reader(csvDataFile, delimiter=",", quoting=csv.QUOTE_NONNUMERIC)
        for row in csvReader:
            extsun_.append([x for x in row if x != ''])
    extsun_ = np.vstack(extsun_)
    # print(f"# Sun is an extended source (see {extsun_coord})...")

    sun_veccs = np.repeat(sun_vecs, extsun_.shape[0], axis=0)
    Zs = np.array([0, 0, 1])
    Us = sun_veccs / np.linalg.norm(sun_veccs, axis=1)[:, np.newaxis]
    Vs = np.cross(Zs, Us)
    Ws = np.cross(Us, Vs)
    Rs = 695700.  # Sun radius, km

    extsun_tiled = np.tile(extsun_, (sun_vecs.shape[0], 1))

    return sun_veccs + Vs * extsun_tiled[:, 0][:, np.newaxis] * Rs + Ws * extsun_tiled[:, 1][:, np.newaxis] * Rs


def get_flux_at_date(shape_model, utc0, path_to_furnsh, albedo1=0.1, Fsun=1361., center='P', point=True):

    F = shape_model.F
    if center == 'V':
        C = shape_model.V
        N = shape_model.VN
    elif center == 'P':
        C = shape_model.P
        N = shape_model.N

    point_sun_vecs = get_sunvec(utc0=utc0, stepet=1, et_linspace=np.linspace(0, 1, 1),
                                path_to_furnsh=path_to_furnsh,
                                target='SUN', frame='MOON_ME', observer='MOON')

    if point:
        # if point Sun
        sun_vecs = point_sun_vecs
        sundir = sun_vecs / np.linalg.norm(sun_vecs)
    else:
        sun_vecs = extended_sun(point_sun_vecs,
                                extsun_coord=f"examples/aux/coordflux_100pts_outline33_centerlast_R1_F1_stdlimbdark.txt")
        sundir = sun_vecs / np.linalg.norm(sun_vecs, axis=1)[:, np.newaxis]

    if center == 'P':
        E = shape_model.get_direct_irradiance(Fsun, sundir)
    elif center == 'V':
        E = shape_model.get_direct_irradiance_at_vertices(Fsun, sundir)

    # # get Moon centered cartesian coordinates of the Sun at date and correct to hr faces centers
    faces_to_sun = point_sun_vecs - C

    # compute incidence angles for visible faces (redundant, many faces are the same, but shouldn't be an issue)
    incidence_angle1 = angle_btw(faces_to_sun, N)
    # compute emission angles to fictious camera above the scene (obs set to zenith)
    emission_angle1 = angle_btw(C, N)
    # compute phase angles
    phase_angle1 = angle_btw(C, faces_to_sun)

    # # get photometry of first bounce
    photom1 = mmpf_mh_boyd2017lpsc(phase=phase_angle1, emission=emission_angle1, incidence=incidence_angle1)

    # # compute radiance out of scatterer
    return E * albedo1 * photom1 * np.pi / Fsun


def render_at_date(meshes, epo_utc, path_to_furnsh, center='P', crs=None):
    """
    Render terrain at epoch
    :param pdir:
    :param meshes: dict
    :param path_to_furnsh:
    :param epo_utc:
    :param outdir:
    :param center:
    :param crs:
    :return:
    """

    input_YYMMGGHHMMSS = datetime.strptime(epo_utc.strip(), '%Y-%m-%d %H:%M:%S.%f')
    format_code = '%Y%m%d%H%M%S'
    date_illum_str = input_YYMMGGHHMMSS.strftime(format_code)
    format_code = '%Y %m %d %H:%M:%S'
    date_illum_spice = input_YYMMGGHHMMSS.strftime(format_code)

    # import hr meshes and build shape_models
    V_st, F_st, N_st, P_st = import_mesh(f"{meshes['stereo']}", get_normals=True, get_centroids=True)
    V, F, N, P = import_mesh(f"{meshes['cart']}", get_normals=True, get_centroids=True)
    shape_model = CgalTrimeshShapeModel(V, F, N)

    # get flux at observer (would be good to just ask for F/V overlapping with meas image)
    flux_at_obs = get_flux_at_date(shape_model, date_illum_spice, path_to_furnsh=path_to_furnsh, center=center)

    plot3d(mesh_path=f"{meshes['cart']}", var_to_plot=flux_at_obs)

    # rasterize results from mesh
    # ---------------------------
    if center == 'V':
        flux_df = pd.DataFrame(np.vstack([V_st[:, 0].ravel(), V_st[:, 1].ravel(), flux_at_obs]).T,
                               columns=['x', 'y', 'flux'])
    elif center == 'P':
        flux_df = pd.DataFrame(np.vstack([P_st[:, 0].ravel(), P_st[:, 1].ravel(), flux_at_obs]).T,
                               columns=['x', 'y', 'flux'])

    flux_df = flux_df.set_index(['y', 'x'])
    ds = flux_df.to_xarray()

    if crs!=None:
        # assign crs
        img_crs = crs
        ds.rio.write_crs(img_crs, inplace=True)

    # interpolate nans
    ds['x'] = ds.x * 1e3
    ds['y'] = ds.y * 1e3
    dsi = ds.interpolate_na(dim="x").interpolate_na(dim="y")

    return dsi, date_illum_str

def render_match_image(pdir, meshes, path_to_furnsh, img_name, epo_utc, meas_path, outdir=None, center='P'):
    """
    Render input terrain at epoch and match observed flux to input image
    :param pdir:
    :param meshes: dict
    :param path_to_furnsh: str
    :param img_name: str
    :param epo_utc: str
    :param meas_path: str
    :param outdir:
    :param center:
    :return: str
    """
    print(f"- Processing {img_name} and clipping to {meas_path}...")

    # define processing dirs
    if outdir == None:
        outdir = f"{pdir}out/"
    os.makedirs(outdir, exist_ok=True)

    # interpolate to NAC nodes
    meas = xr.load_dataarray(meas_path)
    meas = meas.where(meas > 0)

    # cut down to useful (x, y) pairs instead of the whole "rectangle"
    # x = np.linspace(np.min(V_st[:, 0]), np.max(V_st[:, 0]), 1000)
    # y = np.linspace(np.min(V_st[:, 1]), np.max(V_st[:, 1]), 1000)
    x = meas.x.values #*1.e-3
    y = meas.y.values #*1.e-3

    # get full rendering at date
    dsi, date_illum_str = render_at_date(meshes, epo_utc, path_to_furnsh, crs=meas.rio.crs)

    # interp to measured image coordinates
    rendering = dsi.interp(x=x, y=y)
    rendering = rendering.rio.reproject_match(meas, Resampling=Resampling.bilinear,
                                                                          nodata=np.nan).where(meas > 0)

    # apply "exposure factor" (median of ratio) to rendered image
    exposure_factor = (rendering/meas).flux.values
    rendering /= np.nanmedian(exposure_factor)

    # save simulated image to raster
    outraster = f"{outdir}{img_name}_{date_illum_str}.tif"
    # dsi.transpose('y', 'x').rio.to_raster(outraster) # full DEM region
    rendering.sel({'band': 1}).transpose('y', 'x').rio.to_raster(outraster)
    # rendering.rio.to_raster(outraster)
    print(f"- Flux for {img_name} saved to {outraster} (xy resolution = {rendering.rio.resolution()}mpp).")

    return outraster


