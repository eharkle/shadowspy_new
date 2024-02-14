import glob
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr
import json_numpy as json

from matplotlib import pyplot as plt
import colorcet as cc

from shape import CgalTrimeshShapeModel
from mesh_utils import import_mesh
from spice_util import get_sourcevec


def get_values_using_raytracing(field, xgrid, ygrid, shape_model_st):
    import itertools as it
    assert field.ndim == 1 and field.size == shape_model_st.num_faces
    dtype = field.dtype
    d = np.array([0, 0, 1], dtype=np.float64)
    m, n = len(xgrid), len(ygrid)
    grid = np.empty((m, n), dtype=dtype)
    grid[...] = np.nan
    for i, j in it.product(range(m), range(n)):
        x = np.array([xgrid[i], ygrid[j], -200], dtype=dtype)  # z is the camera elevation, fix w.r.t. elevation max/min
        hit = shape_model_st.intersect1(x, d)
        if hit is not None:
            grid[i, j] = field[hit[0]]

    # test to use intersect1_2d (should remove the other loops to make it efficient + very large x matrix...)
    # x = []
    # for i, j in it.product(range(m), range(n)):
    #     x.append(np.array([xgrid[i], ygrid[j], -200], dtype=dtype)) # z is the camera elevation, fix w.r.t. elevation max/min
    # X = np.vstack(x)
    # D = np.tile(d.T, len(x)).reshape(-1,3)
    # hit = shape_model_st.intersect1_2d(X, D)
    # for idx, (i, j) in enumerate(it.product(range(m), range(n))):
    #     if hit[idx] is not None:
    #         grid[i, j] = field[hit[idx]]

    return grid


def plot_ROI_circle(x_ROI, y_ROI, r_ROI, **kwargs):
    theta = np.linspace(0, 2 * np.pi, 301)
    x, y = r_ROI * np.cos(theta) + y_ROI, r_ROI * np.sin(theta) + x_ROI
    plt.plot(x, y, **kwargs)


def plot(data, outpng, extent, cmap=cc.cm.fire, add_roi=False, **kwargs):
    """
    Plot direct illumination map (E_grid) and computed scattered flux (F_grid) from raytracing
    """

    plt.figure(**kwargs)

    for idx, (nam, dat) in enumerate(data.items()):
        plt.subplot(1, len(data.values()), idx+1)
        plt.title(nam) #'direct flux (W/m2)')
        plt.imshow(dat, interpolation='none', extent=extent[idx],
                   vmin=0, cmap=cmap)  # vmax=vmax,
        # plot_ROI_circle(c='cyan', linewidth=1, linestyle='--', zorder=2)
        plt.ylabel('$y$')
        plt.xlabel('$x$')
        plt.colorbar()
        plt.gca().set_aspect('equal')

        if add_roi:
            plot_ROI_circle(x_ROI=add_roi['yc'], y_ROI=add_roi['xc'], r_ROI=add_roi['rc'],
                            c='k', linewidth=2, linestyle='--', zorder=2)

    # plt.subplot(1, 2, 2)
    # plt.title('inner/reflected flux (W/m2)')
    # plt.imshow(data['F_grid'], interpolation='none', extent=extent[1],
    #            vmin=0, cmap=cmap)  # vmax=vmax,
    # plt.ylabel('$y$')
    # plt.xlabel('$x$')
    # plt.colorbar()
    # plt.gca().set_aspect('equal')

    plt.tight_layout()
    plt.savefig(outpng)
    # plt.show()
    # plt.close()

    print(f"- Direct and scattered fluxes plotted to {outpng}.")


def to_geotiff(data, coords, crs, raster_out=None):
    """
    Save computed scattered flux to raster (GeoTiff) with input crs
    """

    # create dataarray (pay attention to dims ordering and units)
    ds = xr.DataArray(
        data=data,
        dims=['y', 'x'],  # y,x is the ordering expected by to_raster
        # coords={'x': coords['x'] * 1e3, 'y': coords['y'][-1:0:-1] * 1e3},  # convert to meters to fit the crs
        coords={'x': coords['x'], 'y': coords['y']},
        attrs={
            '_FillValue': np.nan,
            'units': 'W/m2',
            'crs': crs
        }
    )
    # print(ds)
    # ds.plot()
    # plt.show()

    if raster_out == None:
        return ds
    else:
        ds.rio.to_raster(raster_out)
        print(f"- Flux results saved to {raster_out} (xy resolution = {ds.rio.resolution()}mpp).")

if __name__ == '__main__':

    feature = 'kplo60'
    experiment = f"{feature}_v2"
    base_resolution = 60
    single_decimation = False
    outdir = "out/kplo60_v2/20221230060000/mmpf_mh_boyd2017lpsc"

    input_YYMMGGHHMMSS = datetime(2022, 12, 30, 6, 00, 00)

    required_angle = 3
    if single_decimation:
        decim_maxdist = [0, 50] # 2, 4, 6, 8, 10, 15, 20, 25, 30, 50]
        decim_list = [1, 5] # [int(max(dnmin, dtmp * np.tan(np.deg2rad(angreq)) / (dxa * dno))) for dtmp in decim_maxdist]
    else:
        decim_list = [2, 4, 6, 8, 10, 15, 20, 25, 30, 50]
        decim_maxdist = [int(dtmp * (base_resolution / 1000.) / np.tan(np.deg2rad(required_angle))) for dtmp in
                         decim_list]

    xmin, xmax, ymin, ymax = 18, 79, -65, -5
    extent = [xmin, xmax, ymin, ymax]
    Fsun = 1361  # W/m2

    obsflux_path = f'{outdir}/flux_b{base_resolution}_dn{decim_list[-1]}.json'

    outpng = f'{outdir}/fluxes_b{base_resolution}_dn{decim_list[-1]}.png'
    outraster = f"{outdir}/fluxes_b{base_resolution}_dn{decim_list[-1]}.tif"

    meshpath_st = f"in/{experiment}/b{base_resolution}_dn1_st.ply"
    V_st, F_st, N_st = import_mesh(f"{meshpath_st}", get_normals=True, get_centroids=False)
    shape_model_st = CgalTrimeshShapeModel(V_st, F_st, N_st)

    # set up plot grid
    N_grid = 512  # 1024 # 2048 #
    x = np.linspace(xmin, xmax, N_grid)
    y = np.linspace(ymin, ymax, N_grid)

    # build regular shape model
    meshpath = f"in/{experiment}/b{base_resolution}_dn1.ply"
    V, F, N = import_mesh(f"{meshpath}", get_normals=True, get_centroids=False)
    shape_model = CgalTrimeshShapeModel(V, F, N)

    format_code = '%Y%m%d%H%M%S'
    date_illum_str = input_YYMMGGHHMMSS.strftime(format_code)
    format_code = '%Y %m %d %H:%M:%S'
    date_illum_spice = input_YYMMGGHHMMSS.strftime(format_code)
    utc0 = date_illum_spice
    sun_vecs = get_sourcevec(utc0=utc0, stepet=1, et_linspace=np.linspace(0, 1, 1), path_to_furnsh='aux/simple.furnsh',
                             target='SUN', frame='MOON_ME', observer='MOON')
    E = shape_model.get_direct_irradiance(Fsun, sun_vecs)

    E_grid = get_values_using_raytracing(E, x, y[-1:0:-1], shape_model_st).T

    # read fluxes from each chunk and decimation and sum by roi face to get total at observer per face
    # reduce grid to inner area
    x = np.linspace(xmin, xmax, N_grid)
    y = np.linspace(ymin, ymax, N_grid)

    with open(obsflux_path, 'r') as f:
        scatout = json.load(f)

    incflux_observer = np.vstack([f['hroi_flux'] for f in scatout.values()])
    incflux_observer = pd.DataFrame(incflux_observer).groupby(0, axis=0).sum()
    incflux = incflux_observer.reindex(np.arange(len(F_st))).values[:, 0]
    F_grid = get_values_using_raytracing(incflux, x, y[-1:0:-1], shape_model_st).T

    # field = np.nan_to_num(E_grid) + np.nan_to_num(F_grid)
    # print(F_grid)
    # exit()
    # import pyvista as pv
    # grid = pv.read(meshpath_st)
    # grid.cell_data['flux (W/m2)'] = np.nan
    # grid.cell_data['flux (W/m2)'][incflux_observer.reindex(np.arange(len(F_st))).index.values.astype('int')] = incflux_observer.reindex(np.arange(len(F_st))).values[:,0]
    # grid.plot(show_scalar_bar=True, show_axes=True, cmap=cc.gray) #, clim=[0,5])
    # plt.show()
    # exit()


    # retrieve crs
    moon_sp_crs = xr.open_dataset("aux/illumSun_89S20M_120928180000_001_cut.tif").rio.crs
    to_geotiff(data=F_grid, coords={'x':x, 'y':y}, crs=moon_sp_crs, raster_out=outraster)

    xmin, xmax, ymin, ymax = 40, 50, -40, -30  # -5, 5, -5, 5
    extent_inner = [xmin, xmax, ymin, ymax]
    plot({'E_grid': E_grid, 'F_grid': F_grid}, outpng, extent=[extent, extent_inner])

