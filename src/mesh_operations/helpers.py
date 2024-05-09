import shutil

import numpy as np
import time
import xarray as xr
from tqdm import tqdm

from src.mesh_operations import mesh_generation
from src.mesh_operations.merge_overlapping import merge_inout
from src.mesh_operations import load_mesh
from src.mesh_operations.split_merged import split_merged


def prepare_inner_outer_mesh(tif_path, fartopo_path, extres, max_extension, Rb, tmpdir, meshpath, ext):
    """
    Prepare outer mesh, as cropped from a larger GeoTiff, then stack to inner mesh with transition area

    @param tif_path: str, inner topography GeoTiff
    @param inner_mesh: str, inner mesh path
    @param fartopo_path: str, outer topography GeoTiff
    @param extres: dict, {dist_meters: resol_meters}
    @param ext: str, mesh extension
    @param max_extension: float, maximum distance for outer topography
    @param tmpdir: str
    @param meshpath: str, output full mesh path
    @param Rb: float, planetary radius
    @return: int, str, str
    """

    start = time.time()
    # crop fartopo to box around dem to render
    da = xr.open_dataarray(tif_path)
    bounds = da.rio.bounds()
    demcx, demcy = np.mean([bounds[0], bounds[2]]), np.mean([bounds[1], bounds[3]])

    da_out = xr.open_dataarray(fartopo_path)
    min_resolution = int(round(da_out.rio.resolution()[0],0))

    # Merge inner and outer meshes seamlessly
    # set a couple of layers at 1, 5 and max_extension km ranges
    outer_topos = []
    extres = {ext: max(res, min_resolution) for ext, res in extres.items()
              if ext < max_extension}
    for extension, resol in tqdm(extres.items(), total=len(list(extres.keys())), desc='crop_outer'):
        da_red = da_out.rio.clip_box(minx=demcx-extension, miny=demcy-extension,
                             maxx=demcx+extension, maxy=demcy+extension)
        fartopo_path = f"{tmpdir}LDEM_{int(round(extension,0))}M_outer.tif"
        fartopomesh = fartopo_path.split('.')[0]
        da_red.rio.to_raster(fartopo_path)
        outer_topos.append({resol: f"{tmpdir}LDEM_{int(round(extension,0))}M_outer.tif"})

    # for iter 0, set inner mesh as stacked mesh
    shutil.copy(f"{meshpath}_st{ext}", f"{tmpdir}stacked_st{ext}")
    labels_dict_list = {}
    for idx, resol_dempath in tqdm(enumerate(outer_topos), total=len(outer_topos), desc='stack_meshes'):
        resol = list(resol_dempath.keys())[0]
        dempath = list(resol_dempath.values())[0]

        outer_mesh_resolution = resol
        fartopo_path = dempath
        fartopomesh = fartopo_path.split('.')[0]

        print(f"- Adding {fartopo_path} ({fartopomesh}) at {outer_mesh_resolution}mpp.")

        # ... and outer topography
        mesh_generation.make(outer_mesh_resolution, [1], fartopo_path, out_path=tmpdir, mesh_ext=ext,
                             rescale_fact=1e-3, lonlat0=(0, -90))
        shutil.move(f"{tmpdir}b{outer_mesh_resolution}_dn1_st{ext}", f"{tmpdir}{fartopomesh.split('/')[-1]}_st{ext}")
        print(f"- Meshes generated after {round(time.time() - start, 2)} seconds.")

        stacked_mesh_path = f"{tmpdir}stacked_st{ext}"
        input_totalmesh, labels_dict = merge_inout(load_mesh(stacked_mesh_path),
                                                   load_mesh(f"{tmpdir}{fartopomesh.split('/')[-1]}_st{ext}"),
                                                   output_path=stacked_mesh_path) #, debug=True)
        labels_dict_list[idx] = labels_dict
        print(f"- Meshes merged after {round(time.time() - start, 2)} seconds and saved to {stacked_mesh_path}.")

    start = time.time()
    # Split inner and outer meshes
    len_inner_faces = labels_dict_list[0]['inner']
    inner_mesh_path, outer_mesh_path = split_merged(input_totalmesh, len_inner_faces, meshpath, fartopomesh, ext, Rb)
    print(inner_mesh_path, outer_mesh_path)
    print(f"- Inner+outer meshes generated from merged after {round(time.time() - start, 2)} seconds.")

    return len_inner_faces, inner_mesh_path, outer_mesh_path
