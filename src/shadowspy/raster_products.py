import logging

# from line_profiler_pycharm import profile
from matplotlib import pyplot as plt
import xarray as xr
import rioxarray
from tqdm import tqdm

#@profile
def basic_raster_stats(epo_path_dict, time_step_hours, crs, outdir='.', siteid='', verbose=True):

    # load and stack dataarrays from list
    list_da = []
    for idx, (epo, dsi_path) in tqdm(enumerate(epo_path_dict.items()), total=len(epo_path_dict)):

        da = xr.open_dataset(dsi_path)
        da = da.assign_coords(time=epo)
        da = da.expand_dims(dim="time")
        da['flux'] = da.band_data
        da = da.drop("band_data")
        list_da.append(da)

    ds = xr.combine_by_coords(list_da)
    moon_sp_crs = crs
    ds.rio.write_crs(moon_sp_crs, inplace=True)

    # get cumulative flux (assuming 24H steps for now)
    step_sec = time_step_hours * 3600.
    dssum = (ds * step_sec).sum(dim='time')
    # get max flux
    dsmax = ds.max(dim='time')
    # get average flux
    dsmean = ds.mean(dim='time')

    # save to raster
    epos_utc = list(epo_path_dict.keys())
    try:
        start_time = str(epos_utc[0])
        end_time = str(epos_utc[-1])
    except:
        format_code = '%Y%m%d%H%M%S'
        start_time = epos_utc[0].strftime(format_code)
        end_time = epos_utc[-1].strftime(format_code)

    sumout = f"{outdir}{siteid}_sum_{start_time}_{end_time}.tif"
    dssum.flux.rio.to_raster(sumout)
    logging.info(f"- Cumulative flux "
                 #f"over {list(dsi_list.keys())[0]} to {list(dsi_list.keys())[-1]} "
                 f"saved to {sumout}.")

    maxout = f"{outdir}{siteid}_max_{start_time}_{end_time}.tif"
    dsmax.flux.rio.to_raster(maxout)
    logging.info(f"- Maximum flux "
                 #f"over {list(dsi_list.keys())[0]} to {list(dsi_list.keys())[-1]} "
                 f"saved to {maxout}.")

    meanout = f"{outdir}{siteid}_mean_{start_time}_{end_time}.tif"
    dsmean.flux.rio.to_raster(meanout)
    logging.info(f"- Average flux "
                 #f"over {list(dsi_list.keys())[0]} to {list(dsi_list.keys())[-1]} "
                 f"saved to {meanout}.")

    # plot statistics
    fig, axes = plt.subplots(1, 3, figsize=(26, 6))
    dssum.flux.plot(ax=axes[0], robust=True)
    axes[0].set_title(r'Sum (J/m$^2$)')
    dsmax.flux.plot(ax=axes[1], robust=True)
    axes[1].set_title(r'Max (J/m$^2$/s)')
    dsmean.flux.plot(ax=axes[2], robust=True)
    axes[2].set_title(r'Mean (J/m$^2$/s)')
    plt.suptitle(f'Statistics of solar flux at {siteid} between {start_time} and {end_time}.')
    pngout = f"{outdir}{siteid}_stats_{start_time}_{end_time}.png"
    plt.savefig(pngout) # 70%+ of time is spent here
    # plt.show()