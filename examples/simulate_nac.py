import glob
import os
import shutil
import time
import pandas as pd

from examples.download_kernels import download_kernels
from src import prepare_meshes
from src.image_util import read_img_properties
from src.render_dem import render_match_image

if __name__ == '__main__':

    # compute direct flux from the Sun
    Fsun = 1361  # W/m2
    Rb = 1737.4 # km
    base_resolution = 20
    root = "examples/"
    os.makedirs(root, exist_ok=True)

    # download kernels
    download_kernels()

    # Elevation/DEM GTiff input
    indir = f"{root}aux/"
    tif_path = f'{indir}ldem_6.tif'#
    meshpath = tif_path.split('.')[0]

    # prepare mesh of the input dem
    start = time.time()
    print(f"- Computing trimesh for {tif_path}...")

    # regular delauney mesh
    ext = '.vtk'
    prepare_meshes.make(base_resolution, [1], tif_path, out_path=root, mesh_ext=ext)
    shutil.move(f"{root}b{base_resolution}_dn1{ext}", f"{meshpath}{ext}")
    shutil.move(f"{root}b{base_resolution}_dn1_st{ext}", f"{meshpath}_st{ext}")
    print(f"- Meshes generated after {round(time.time() - start, 2)} seconds.")

    # open index
    lnac_index = f"{indir}CUMINDEX_LROC.TAB"
    cumindex = pd.read_csv(lnac_index, index_col=None)

    # get list of images from mapprojected folder
    imgs = glob.glob(f"{indir}M1*E_map.tif")[:]
    imgs_names = [f.split('/')[-1].split('_map')[0] for f in imgs]
    imgs_nam_epo_path = read_img_properties(imgs_names, cumindex)

    imgs_nam_epo_path['meas_path'] = [f"{indir}{img}_map.tif"
                                      for img in imgs_nam_epo_path.PRODUCT_ID.values]
    print(f"- {len(imgs_nam_epo_path)} images found in path. Rendering input DEM.")

    for idx, row in imgs_nam_epo_path.iterrows():
        render_match_image(root, meshes={'stereo': f"{meshpath}_st{ext}", 'cart': f"{meshpath}{ext}"},
                       path_to_furnsh=f"{indir}simple.furnsh",
                       img_name=row[0], epo_utc=row[1], meas_path=row[2],
                       center='P')
