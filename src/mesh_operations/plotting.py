import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import rioxarray

def plot_transition(vertices_in, faces_in, vertices_out, faces_out, transition_vertices, transition_faces):
    pass

def rasterize_grid(gridded_data, bounds, crs, to_geotiff=None):

    xmin, xmax, ymin, ymax = bounds

    # Create coordinate arrays based on the bounds and grid size
    x_coords = np.linspace(xmin, xmax, gridded_data.shape[1])
    y_coords = np.linspace(ymax, ymin, gridded_data.shape[0])  # Reverse y for correct orientation

    # Create xarray DataArray
    da = xr.DataArray(
        gridded_data,
        coords={"y": y_coords, "x": x_coords},
        dims=("y", "x"),
    )

    # Assign CRS to the DataArray
    da = da.rio.write_crs(crs)

    if to_geotiff != None:
        # Save as GeoTIFF
        da.rio.to_raster(to_geotiff, compress="LZW")

    return da

if __name__ == '__main__':
    # Example data (replace with your actual data)
    example_gridded_data = np.random.rand(512, 512)  # Replace with your (512, 512) grid
    xmin, xmax, ymin, ymax = -65, -25, -20, 20

    da = rasterize_grid(example_gridded_data, (xmin, xmax, ymin, ymax), crs='WGS84')
    da.plot(robust=True)
    plt.show()