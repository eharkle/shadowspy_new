import os
import subprocess

# Scope: 
# Download the spice kernels needed
# NOTE: At the moment this script works if either wget or curl are installed in the system

def download_kernels():

    # From PDS
    todownload = ['spk/de421.bsp',
                  'fk/moon_assoc_pa.tf',
                  'fk/moon_080317.tf',
                  'pck/moon_pa_de421_1900_2050.bpc',
                  'pck/pck00010.tpc',
                  'lsk/naif0012.tls',
                  ]

    for f in todownload:
        fname = os.path.basename(f)
        if not os.path.exists(f'examples/kernels/{fname}'):
            try:
                subprocess.run(
                    f'wget -P examples/kernels/  https://naif.jpl.nasa.gov/pub/naif/pds/data/lro-l-spice-6-v1.0/lrosp_1000/data/{f}',
                    shell=True)
            except:
                subprocess.run(
                    f'curl https://naif.jpl.nasa.gov/pub/naif/pds/data/lro-l-spice-6-v1.0/lrosp_1000/data/{f} -o examples/kernels/{fname}',
                    shell=True)
