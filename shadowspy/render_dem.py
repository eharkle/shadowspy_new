import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd
import geopandas as gpd
from matplotlib import pyplot as plt

from shadowspy.import_mesh import import_mesh
from shadowspy.mesh_util import crop_mesh
from shadowspy.spice_util import get_sourcevec
from shadowspy.shape import CgalTrimeshShapeModel  # , EmbreeTrimeshShapeModel
import xarray as xr
from rasterio.enums import Resampling

from shadowspy.photometry import mmpf_mh_boyd2017lpsc
from shadowspy.math_util import angle_btw


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


def get_flux_at_date(shape_model, utc0, path_to_furnsh, albedo1=0.1, source='SUN', inc_flux=1361., center='P',
                     point=True, basemesh=None, return_irradiance=False):
    if center == 'V':
        C = shape_model.V
        N = shape_model.VN
    elif center == 'P':
        C = shape_model.P
        N = shape_model.N

    point_source_vecs = get_sourcevec(utc0=utc0, stepet=1, et_linspace=np.linspace(0, 1, 1),
                                   path_to_furnsh=path_to_furnsh,
                                   target=source, frame='MOON_ME', observer='MOON')

    if point:
        # if point Sun
        source_vecs = point_source_vecs
        sourcedir = source_vecs / np.linalg.norm(source_vecs)
    else:
        if source == 'SUN':
            source_vecs = extended_sun(point_source_vecs,
                                extsun_coord=f"examples/aux/coordflux_100pts_outline33_centerlast_R1_F1_stdlimbdark.txt")
        elif source == 'EARTH':
            source_vecs = extended_sun(point_source_vecs,
                                extsun_coord=f"examples/aux/Source2D_Earth_100pts_centerlast.inc")
        else:
            logging.error(f"* Only SUN and EARTH set as possible light-sources. User asked for {source}. Exit.")
            exit()

        sourcedir = source_vecs / np.linalg.norm(source_vecs, axis=1)[:, np.newaxis]

    if center == 'P':
        E = shape_model.get_direct_irradiance(inc_flux, sourcedir, basemesh=basemesh)
    elif center == 'V':
        E = shape_model.get_direct_irradiance_at_vertices(inc_flux, sourcedir, basemesh=basemesh)

    if return_irradiance:
        return E

    # # get Moon centered cartesian coordinates of the Sun at date and correct to hr faces centers
    faces_to_sun = point_source_vecs - C

    # compute incidence angles for visible faces (redundant, many faces are the same, but shouldn't be an issue)
    incidence_angle1 = angle_btw(faces_to_sun, N)
    # compute emission angles to fictious camera above the scene (obs set to zenith)
    emission_angle1 = angle_btw(C, N)
    # compute phase angles
    phase_angle1 = angle_btw(C, faces_to_sun)

    # # get photometry of first bounce
    photom1 = mmpf_mh_boyd2017lpsc(phase=phase_angle1, emission=emission_angle1, incidence=incidence_angle1)

    # # compute radiance out of scatterer
    return E * albedo1 * photom1 * np.pi / inc_flux


def render_at_date(meshes, epo_utc, path_to_furnsh, center='P', crs=None, dem_mask=None, source='SUN',
                   inc_flux=1361, basemesh_path=None, show=False, point=True, return_irradiance=False):
    """
    Render terrain at epoch
    :param pdir:
    :param meshes: dict
    :param path_to_furnsh:
    :param epo_utc:
    :param outdir:
    :param center:
    :param crs:
    :param dem_mask: GeoDataFrame, polygon of region to crop
    :return:
    """

    input_YYMMGGHHMMSS = datetime.strptime(epo_utc.strip(), '%Y-%m-%d %H:%M:%S.%f')
    format_code = '%Y %m %d %H:%M:%S'
    date_illum_spice = input_YYMMGGHHMMSS.strftime(format_code)

    # check if DEM needs to be cropped (e.g., to fit image)
    if isinstance(dem_mask, gpd.GeoDataFrame):
        print(f"- Cropping DEM to {dem_mask}")
        meshes_cropped = {}
        meshes_path = ('/').join(meshes['stereo'].split('/')[:-1])
        meshes_cropped['stereo'] = f"{meshes_path}/cropped_st.ply"
        meshes_cropped['cart'] = f"{meshes_path}/cropped.ply"

        crop_mesh(dem_mask, meshes, mask=dem_mask, meshes_cropped=meshes_cropped)

        V_st, F_st, N_st, P_st = import_mesh(f"{meshes_cropped['stereo']}", get_normals=True, get_centroids=True)
        V, F, N, P = import_mesh(meshes_cropped['cart'], get_normals=True, get_centroids=True)
    else:
        # import hr meshes and build shape_models
        V_st, F_st, N_st, P_st = import_mesh(f"{meshes['stereo']}", get_normals=True, get_centroids=True)
        V, F, N, P = import_mesh(f"{meshes['cart']}", get_normals=True, get_centroids=True)
        meshes_cropped = meshes

    shape_model = CgalTrimeshShapeModel(V, F, N)

    if basemesh_path != None:
        V_ds, F_ds, N_ds, P_ds = import_mesh(basemesh_path, get_normals=True, get_centroids=True)
        basemesh = CgalTrimeshShapeModel(V_ds, F_ds, N_ds)
    else:
        basemesh = None

    # get flux at observer (would be good to just ask for F/V overlapping with meas image)
    flux_at_obs = get_flux_at_date(shape_model, date_illum_spice, path_to_furnsh=path_to_furnsh,
                                   source=source, inc_flux=inc_flux,
                                   center=center, point=point, basemesh=basemesh, return_irradiance=return_irradiance)

    if show:
        # plot3d(mesh_path=f"{meshes['cart']}", var_to_plot=flux_at_obs)
        plot3d(mesh_path=meshes_cropped['stereo'], var_to_plot=flux_at_obs)

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

    if crs != None:
        # assign crs
        img_crs = crs
        ds.rio.write_crs(img_crs, inplace=True)

    # interpolate nans
    ds['x'] = ds.x * 1e3
    ds['y'] = ds.y * 1e3
    dsi = ds.interpolate_na(dim="x").interpolate_na(dim="y")

    return dsi, epo_utc


def irradiance_at_date(meshes, epo_utc, path_to_furnsh, center='P', crs=None, dem_mask=None, source='SUN',
                       inc_flux=1361, basemesh_path=None, show=False, point=True, return_irradiance=True):
    """
    Get terrain irradiance at epoch
    :param inc_flux:
    :param basemesh_path:
    :param show:
    :param point:
    :param pdir:
    :param meshes: dict
    :param path_to_furnsh:
    :param epo_utc:
    :param outdir:
    :param center: str, can take values P or V
    :param crs: str
    :param dem_mask: GeoDataFrame, polygon of region to crop
    :param return_irradiance: bool, must be True
    :return:
    """
    if not return_irradiance:
        logging.error("* Either set return_irradiance=True, or else call render_at_date.")

    return render_at_date(meshes, epo_utc, path_to_furnsh, center, crs, dem_mask,source,
                          inc_flux, basemesh_path, show, point, return_irradiance)


def render_match_image(pdir, meshes, path_to_furnsh, img_name, epo_utc,
                       meas_path, outdir=None, center='P', basemesh=None, point=True):
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
    meas = meas.where(meas >= 0)

    ############# simulate smaller image
    # import shapely
    # geometries = shapely.box(67400, 121800, 68400, 122200)
    # meas = meas.rio.clip([geometries])
    #############

    # raster outer shape to polygon
    ds = (meas.coarsen(x=1, boundary="trim").mean(skipna=True).
          coarsen(y=1, boundary="trim").mean(skipna=True))
    df = ds.to_dataframe().reset_index().loc[:, ['x', 'y']] * 1e-3
    meas_outer_poly = gpd.GeoDataFrame(geometry=gpd.points_from_xy(df.x, df.y))
    meas_outer_poly = meas_outer_poly.dropna(axis=0).dissolve().convex_hull
    meas_outer_poly = gpd.GeoDataFrame({'geometry': meas_outer_poly.buffer(50, join_style=2)})

    # get full rendering at date
    dsi, date_illum_str = render_at_date(meshes, epo_utc, path_to_furnsh, crs=meas.rio.crs,
                                         dem_mask=meas_outer_poly, center=center,
                                         basemesh_path=basemesh, point=point
                                         )

    # interp to measured image coordinates
    rendering = dsi.rio.reproject_match(meas, Resampling=Resampling.bilinear,
                                        nodata=np.nan)

    # apply "exposure factor" (median of ratio) to rendered image
    exposure_factor = rendering.flux.values / meas.sel({'band': 1}).values
    exposure_factor = np.median(exposure_factor.ravel()[~np.isnan(exposure_factor.ravel())])

    if exposure_factor > 0:
        rendering /= exposure_factor
    else:  # weird cases
        max_ratio = rendering.max() / meas.sel({'band': 1}).max()
        rendering /= max_ratio
        print(f"# Exposure=={exposure_factor}: possible issue or mainly shadowed image (?). "
              f"Normalizing with max_ratio={max_ratio.flux.values}.")

    # save simulated image to raster
    outraster = f"{outdir}{img_name}_{date_illum_str}.tif"
    rendering.transpose('y', 'x').rio.to_raster(outraster)

    #####
    dsi.transpose('y', 'x').rio.to_raster(outraster) # full DEM region
    rendering.flux.plot(robust=True)
    plt.show()
    meas.plot(robust=True)
    plt.show()
    # rendering.rio.to_raster(outraster)
    #####

    print(f"- Flux for {img_name} saved to {outraster} (xy resolution = {rendering.rio.resolution()}mpp).")

    return outraster
