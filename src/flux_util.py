import xarray as xr
# import datetime

def get_Fsun(filin, epoch, wavelength=None,  ):

    ds = xr.open_dataset(filin)
    # avoid issue with pre-1670 epochs
    ds_red = ds.isel(time=slice(100, 1000))  # reduce ds to useful epochs

    # convert cftimes to datetime
    datetimeindex = ds_red.indexes['time'].to_datetimeindex()
    ds_red['time'] = datetimeindex
    ds_red = ds_red.sel(time=epoch, method='nearest')

    # select range of epochs (ds_red does not accept "nearest")
    # epo0 = datetime.datetime.strptime(epoch, '%Y-%m-%d %H:%M:%S.%f')
    # epo1 = (epo0 + datetime.timedelta(days=366))

    if isinstance(wavelength, list):
        ds_res = ds_red.sel(  # time=slice(epo0.strftime('%Y-%m-%d'), epo1.strftime('%Y-%m-%d')),  #
            wavelength=slice(wavelength[0], wavelength[-1])).SSI.sum(dim='wavelength')
    else:
        ds_res = ds_red.sel(  # time=epo, #slice("2000-01-01", "2022-01-02"),
            wavelength=wavelength).SSI
    # plt.title("Solar Flux @1AU 115-320 nm (W/m2) ")
    # plt.show()
    # exit()
    return ds_res.time.values, ds_res.values