"""
maketensor.py

Takes one CCD sequence of DIFFEXP images,
finds one planted object across frames,
extracts 200 x 200 cutouts,
and saves them as a tensor.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from pathlib import Path

import sys
sys.path.append('/arc/home/malvnair/trippy')
from trippy import psf
from pathlib import Path



# User input

INPUT_PATH = input("Enter sequence directory or FITS file: ")
CCD = input("Enter CCD number (def=15): ")
N_FRAMES = input("Enter number of frames (def=10): ")
HALF_SIZE = input("Enter cutout size (def=100): ")


# default to test

if INPUT_PATH == "":
    INPUT_PATH = "/arc/projects/classy/dbimages/2773118/fk2773118p.fits"

if CCD == "":
    CCD = 15
else:
    CCD = int(CCD)

if N_FRAMES == "":
    N_FRAMES = 10
else:
    N_FRAMES = int(N_FRAMES)

if HALF_SIZE == "":
    HALF_SIZE = 100
else:
    HALF_SIZE = int(HALF_SIZE)
    

INPUT_PATH = Path(INPUT_PATH)





# SWITCH SETTTINGS!

MODE = "fake"             # real or fake





OUT_DIR = Path("./tensors")
OUT_DIR.mkdir(exist_ok=True)

#arcseconds/pixel for hsc
PIXEL_SCALE = 0.185  # get from header

HOURS_PER_DAY = 24.0

PLANT_COLS = [
    "index", "ra", "dec", "x", "y",
    "rate", "angle", "rate_ra", "rate_dec",
    "mag", "psf_amp", "g_i"
]



# gets matching plantlist filename
def get_plant_file(fits_file, CCD):

    name = fits_file.stem

    if name.startswith("DIFFEXP"):

        # DIFFEXP-2773082-2773118-15.fits
        parts = name.split("-")

        image_id = parts[1]
        ref_id = parts[2]
        CCD = int(parts[3])

        plant_file = (
            fits_file.parent /
            f"{image_id}p{CCD}-{ref_id}p{CCD}.plantList"
        )

    else:

        # fk2773118p.fits or 2773118p.fits
        image_id = (
            name
            .replace("fk", "")
            .replace("p", "")
        )

        plant_file = (
            fits_file.parent /
            f"ccd{CCD}" /
            f"{image_id}p{CCD}.plantList"
        )

    return plant_file, image_id, CCD



# gets matching psf filename
def get_psf_file(fits_file, CCD):

    name = fits_file.stem

    if name.startswith("DIFFEXP"):

        # DIFFEXP-2773082-2773118-15.fits
        parts = name.split("-")

        image_id = parts[1]
        CCD = int(parts[3])

    else:

        # fk2773118p.fits or 2773118p.fits
        image_id = (
            name
            .replace("fk", "")
            .replace("p", "")
        )

    psf_file = (
        Path("/arc/projects/classy/dbimages") /
        image_id /
        f"ccd{CCD}" /
        f"{image_id}p{CCD}.psf.fits"
    )

    return psf_file



#find all the fits files with the matching pattern and return
if INPUT_PATH.is_file():

    fits_files = [INPUT_PATH]

else:

    fits_files = sorted(
        INPUT_PATH.glob(f"DIFFEXP-*-{ref_id}-{CCD}.fits")
    )


# only take user input of files
fits_files = fits_files[:N_FRAMES]

for f in fits_files:
    print(f.name)

# Initialize empy lists to load images and plantlists
images = []
mjds = []
plant_tables = []
image_ids = []
exptimes = []
psf_files = []



for fits_file in fits_files:
    
    plant_file, image_id, CCD = get_plant_file(fits_file, CCD)
    image_ids.append(image_id)
    
    psf_file = get_psf_file(fits_file, CCD)

    if not plant_file.exists():
        sys.exit(f"Missing plantList: {plant_file}")

    if not psf_file.exists():
        sys.exit(f"Missing PSF file: {psf_file}")

    print(f"\nFITS = {fits_file.name}")
    print(f"plantList = {plant_file}")
    print(f"PSF = {psf_file}")

    with fits.open(fits_file) as hdul:

        name = fits_file.stem

        if name.startswith("DIFFEXP"):

            # DIFFEXP files have image data in HDU 1
            hdu_used = 1

        else:

            # fk dbimages have image data in HDU 0
            # full fk mosaic files have PRIMARY in HDU 0, so CCD 15 is HDU 16
            # non-fk dbimage mosaic files are handled the same as fk mosaic files
            hdu_used = CCD + 1

        # get image array and convert to float
        image = hdul[hdu_used].data.astype(float)

        if "EXPTIME" in hdul[0].header:
            EXPTIME = hdul[0].header["EXPTIME"]
        elif "EXPTIME" in hdul[hdu_used].header:
            EXPTIME = hdul[hdu_used].header["EXPTIME"]
        else:
            EXPTIME = np.nan
            print("No EXPTIME found")
        
        # cent_time = MJD-OBS + exptime/2./(3600.*24.)

        if "MJD-OBS" in hdul[0].header:
            # read modified julian date
            mjd = float(hdul[0].header["MJD-OBS"])
        elif "MJD-OBS" in hdul[hdu_used].header:
            # read modified julian date
            mjd = float(hdul[hdu_used].header["MJD-OBS"])
        else:
            mjd = np.nan
            print("No MJD-OBS found")

    df = pd.read_csv(
        plant_file,
        sep=r"\s+",
        comment="#",
        names=PLANT_COLS,
        dtype=float
    )

    # convert ids so they not decimals
    df["index"] = df["index"].astype(int)

    images.append(image)
    mjds.append(mjd)
    plant_tables.append(df)
    exptimes.append(EXPTIME)
    psf_files.append(psf_file)

    print(f"HDU = {hdu_used}")
    print(f"Image shape = {image.shape}")
    print(f"MJD = {mjd}")
    print(f"Objects = {len(df)}")

# is any nan?
for i in range(len(mjds)):
    if np.isnan(mjds[i]):
        print(f"MJD-OBS missing in frame {i}")



# Find object IDs that appear in every frame
first_table = plant_tables[0]
common_objects = []

for obj in first_table["index"]:
    found_in_all = True
    for df in plant_tables[1:]:
        if obj not in df["index"].values:
            found_in_all = False

    if found_in_all:
        common_objects.append(obj)

print(f"\nObjects common to all frames: {len(common_objects)}")

if len(common_objects) == 0:
    sys.exit("No common objects found across all frames.")


# Choose possible objects (full info) 
possible = first_table[
    first_table["index"].isin(common_objects)
].copy()

height, width = images[0].shape

safe_indices = set(common_objects)

for df in plant_tables:

    safe_here = df[
        (df["x"] >= HALF_SIZE) &
        (df["x"] < width - HALF_SIZE) &
        (df["y"] >= HALF_SIZE) &
        (df["y"] < height - HALF_SIZE)
    ]

    safe_indices = safe_indices.intersection(set(safe_here["index"].values))

safe = possible[possible["index"].isin(safe_indices)].copy()

if safe.empty:
    sys.exit("boop")

safe = safe.sort_values("mag")
# brightest star for now
chosen = safe.iloc[0]

obj_index = int(chosen["index"])

print("\nChosen object:")
print(f"index = {obj_index}")
print(f"mag = {chosen['mag']:.2f}")
print(f"rate = {chosen['rate']:.3f} arcsec/hr")
print(f"angle = {chosen['angle']:.2f}")
print(f"rate_ra = {chosen['rate_ra']:.3f} arcsec/hr")
print(f"rate_dec = {chosen['rate_dec']:.3f} arcsec/hr")


# Cutout function

def get_cutout(image, x, y, half_size):

    height, width = image.shape

    cx = int(round(x))
    cy = int(round(y))

    x0 = cx - half_size
    x1 = cx + half_size

    y0 = cy - half_size
    y1 = cy + half_size

    if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
        return None

    cutout = image[y0:y1, x0:x1].copy()
    cutout[np.isnan(cutout)] = np.nan
    return cutout



from astropy.modeling import functional_models as apmodels
import numpy as np
import sys



# fake injection fucntion

def inject_fake(image, x, y, rate, angle, exptime, psf_file):

    new_image = image.copy().astype(float)

    mpsf = psf.modelPSF(restore=str(psf_file))

    # trailed
    mpsf.line(
        rate,
        angle,
        exptime / 3600.0,
        pixScale=PIXEL_SCALE,
        useLookupTable=True
    )
    
    # plant the fake one
    # normalize the mag and adding noise (look at CFHTMOP)
    fake = mpsf.plant(
        np.array([x]),
        np.array([y]),
        # ask wes whats a good value?
        # use zerpoint 
        np.array([100.0]),
        image * 0.0,
        useLinePSF=True,
        returnModel=True,
        gain=1.0,
        addNoise=False,
        verbose=False
    )

    new_image += fake

    return new_image

# Get positions and cutouts

cutouts = []
positions = []
plant_positions = []

x_start = chosen["x"]
y_start = chosen["y"]

rate_ra = chosen["rate_ra"]
rate_dec = chosen["rate_dec"]

mjd0 = mjds[0]

print("\nTensor cutouts:")
print("frame   dt_hr    x_used   y_used   "
    "dx_used  dy_used   dx_exp   dy_exp   dx_plant dy_plant"
)

# loop through each image/frame
for i in range(len(images)):

    image = images[i]
    df = plant_tables[i]
    mjd = mjds[i]
    EXPTIME = exptimes[i]
    psf_file = psf_files[i]

    plant_row = df[df["index"] == obj_index].iloc[0]
    x_plant = plant_row["x"]
    y_plant = plant_row["y"]

    dt_hours = (mjd - mjd0) * HOURS_PER_DAY

    dx_expected = (rate_ra * dt_hours / PIXEL_SCALE)
    dy_expected =  -(rate_dec * dt_hours / PIXEL_SCALE)

    # use position from plantlist
    if MODE == "real":

        x = x_plant
        y = y_plant
    
    # calc where object should be and inject fake
    elif MODE == "fake":

        x = x_start + dx_expected
        y = y_start + dy_expected

        image = inject_fake(image,x,y,chosen["rate"],chosen["angle"],EXPTIME,psf_file)

    else:
        sys.exit("MODE must be real or fake")

    cutout = get_cutout(image, x, y, HALF_SIZE)

    if cutout is None:
        sys.exit(f"Object too close to edge in frame {i}")

    cutouts.append(cutout)
    positions.append((x, y))
    plant_positions.append((x_plant, y_plant))

    dx_used = x - x_start
    dy_used = y - y_start

    dx_plant = x_plant - x_start
    dy_plant = y_plant - y_start

    print(
        f"{i:5d} {dt_hours:7.3f} "
        f"{x:8.2f} {y:8.2f} "
        f"{dx_used:8.3f} {dy_used:8.3f} "
        f"{dx_expected:8.3f} {dy_expected:8.3f} "
        f"{dx_plant:8.3f} {dy_plant:8.3f}"
    )

# Make tensor

clean_cutouts = []

for cutout in cutouts:

    c = cutout.copy()


    clean_cutouts.append(c)

tensor = np.stack(clean_cutouts, axis=0)

print(f"\nshape = {tensor.shape}")
print(f"min = {np.min(tensor):.2f}")
print(f"max = {np.max(tensor):.2f}")
print(f"mean = {np.mean(tensor):.2f}")


# Save tensor

out_npz = OUT_DIR / f"tensor_CCD{CCD}_OBJ{obj_index}_{MODE}.npz"

np.savez_compressed(
    out_npz,
    tensor=tensor,
    positions=np.array(positions),
    plant_positions=np.array(plant_positions),
    mjds=np.array(mjds),
    obj_index=obj_index,
    mag=chosen["mag"],
    rate=chosen["rate"],
    rate_ra=chosen["rate_ra"],
    rate_dec=chosen["rate_dec"],
)

print()
print(f"Saved tensor: {out_npz}")


# plot tensor frames


n = tensor.shape[0]

cols = 5
rows = int(np.ceil(n / cols))

vmin = np.percentile(tensor, 1)
vmax = np.percentile(tensor, 99)

fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
axes = np.array(axes).reshape(rows, cols)

for i in range(rows * cols):

    ax = axes.flat[i]

    if i < n:

        ax.imshow(
            tensor[i],
            origin="lower",
            cmap="gray",
            vmin=vmin,
            vmax=vmax
        )

        ax.plot(HALF_SIZE, HALF_SIZE, "+", color="red", markersize=12)

        x, y = positions[i]
        xp, yp = plant_positions[i]

        ax.set_title(
            f"frame {i}\nused=({x:.1f},{y:.1f})\nplant=({xp:.1f},{yp:.1f})",
            fontsize=7
        )

    ax.axis("off")

plt.suptitle(
    f"Object {obj_index}, mag={chosen['mag']:.2f}, tensor shape={tensor.shape}"
)

plt.tight_layout()

out_png = OUT_DIR / f"tensor_CCD{CCD}_OBJ{obj_index}_{MODE}.png"

plt.savefig(out_png, dpi=150)
plt.show()
