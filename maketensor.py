"""
maketensor.py

Takes one CCD sequence,
finds brightest planted object across frames,
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

fits_or_dir = input(
    "Enter FITS file or sequence directory "
    "(press enter for dbimages mode): "
)

REF_ID    = input("Enter reference ID (def=2773118): ")
CCD       = input("Enter CCD number (def=15): ")
N_FRAMES  = input("Enter number of frames (def=10): ")
HALF_SIZE = input("Enter cutout size (def=100): ")
PSF_FILE  = input(
    "Enter PSF file "
    "(leave blank to auto-find/skip): "
)


# default to test

if REF_ID == "":
    REF_ID = "2773118"

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

# None = auto-find (dbimages) or skip (warp)
if PSF_FILE == "":
    PSF_FILE = None



    
    

# SWITCH SETTINGS!

MODE   = "real"          # real or fake
SOURCE = "warp"          # warp, dbimages_fk, or dbimages







# pixel offsets applied to fake injection position
X_OFFSET = 5.0
Y_OFFSET = 5.0

# use this when user input for source is dbimages paths
DB_ROOT = Path("/arc/projects/classy/dbimages")

# filled automatically from discover_sequence or exact file mode
IMAGE_IDS = []


OUT_DIR = Path("./tensors")
OUT_DIR.mkdir(exist_ok=True)

# arcseconds/pixel for hsc
PIXEL_SCALE = 0.185  # get from header

HOURS_PER_DAY = 24.0

PLANT_COLS = [
    "index", "ra", "dec", "x", "y",
    "rate", "angle", "rate_ra", "rate_dec",
    "mag", "psf_amp", "g_i"
]




################### Helper functions for finding files!!!@
##########



# find plantlist in either the ccd subfolder or the imageid folder
def find_plantlist(image_id, ccd):
    path_ccd = DB_ROOT / image_id / f"ccd{ccd}" / f"{image_id}p{ccd}.plantList"
    path_top = DB_ROOT / image_id / f"{image_id}p{ccd}.plantList"
    if path_ccd.exists():
        return path_ccd
    if path_top.exists():
        return path_top
    sys.exit("plantlist not found.")


# find psf in either the ccd subfolder or the imageid folder
def find_psf_file(image_id, ccd):
    path_ccd = DB_ROOT / image_id / f"ccd{ccd}" / f"{image_id}p{ccd}.psf.fits"
    path_top = DB_ROOT / image_id / f"{image_id}p{ccd}.psf.fits"
    if path_ccd.exists():
        return path_ccd
    if path_top.exists():
        return path_top
    return None


# go from anchor_id to discover the full consecutive sequence
def discover_sequence(anchor_id_str):
    anchor = int(anchor_id_str)
    found  = []

    # go downward from anchor
    image_id = anchor
    while (DB_ROOT / str(image_id)).is_dir():
        found.append(image_id)
        image_id -= 1

    # go upward from anchor+1
    image_id = anchor + 1
    while (DB_ROOT / str(image_id)).is_dir():
        found.append(image_id)
        image_id += 1

    found.sort()
    return [str(x) for x in found]





#################### figure out what the user gave us

p = Path(fits_or_dir.strip()) if fits_or_dir.strip() else None

# DO THIS TO AVOID NAMEERROR
SEQ_DIR = None

if p is not None and p.is_file() and p.suffix in (".fits", ".fit"):

    # exact fits file get source from filename
    stem = p.stem
    if stem.startswith("DIFFEXP"):
        SOURCE  = "warp"
        # .parent gets the folder with that file
        SEQ_DIR = p.parent
        # get REF_ID and CCD from filename
        parts = stem.split("-")
        if len(parts) >= 4:
            REF_ID = parts[2]
            CCD = int(parts[3])
            
    # get the image sequence, call function
    elif stem.startswith("fk"):
        SOURCE    = "dbimages_fk"
        start_id  = stem[2:].rstrip("p")
        IMAGE_IDS = discover_sequence(start_id)
    else:
        SOURCE    = "dbimages"
        start_id  = stem.rstrip("p")
        IMAGE_IDS = discover_sequence(start_id)


elif p is not None and p.is_dir():

    # sequence directory is warp 
    SOURCE  = "warp"
    SEQ_DIR = p
else:

    # dbimages mode, ask source type, anchor on ref-id
    src_type = input("Enter source type [fk / nonfk]: ").strip().lower()

    if src_type == "fk":
        SOURCE = "dbimages_fk"
    else:
        SOURCE = "dbimages"

    # use ref-id as anchor and get full consecutibe sequence from there
    IMAGE_IDS = discover_sequence(REF_ID)


    
    
    
    
    
 ####### build up fits sequence to use!


# fiind fits files based on source
if SOURCE == "warp":

    # find all the fits files with the matching pattern and return
    fits_files = sorted(
        SEQ_DIR.glob(f"DIFFEXP-*-{REF_ID}-{CCD}.fits")
    )

    # in case you cant find any
    if len(fits_files) == 0:
        sys.exit("No DIFFEXP files found")

    # only take user input of files
    fits_files = fits_files[:N_FRAMES]

elif SOURCE in ("dbimages_fk", "dbimages"):

    fits_files = []
    for image_id in IMAGE_IDS:
        if SOURCE == "dbimages_fk":
            # mosaic fk file no ccd subfolder, CCD selected by HDU
            fits_path = DB_ROOT / image_id / f"fk{image_id}p.fits"
        else:
            # mosaic non-fk file no ccd subfolder, ccd selected by HDU
            fits_path = DB_ROOT / image_id / f"{image_id}p.fits"
        fits_files.append(fits_path)

    # keep only first few frames
    fits_files = fits_files[:N_FRAMES]

else:
    sys.exit("Must be warp, dbimages_fk, or dbimages")

  
    
# make sure its consecutive
for f in fits_files:
    print(f.name)
    

    
    
    
    
######## Get data from the fits files
    
    
# initialize empty lists to load images and plantlists
images      = []
mjds        = []
cent_times  = []
plant_tables = []
image_ids   = []
exptimes    = []
psf_files   = []
zeropoints  = []
gains       = []



for fits_file in fits_files:

    # split up the name!
    name  = fits_file.stem
    parts = name.split("-")

    # extract image_id depending on source
    if SOURCE == "warp":
        image_id = parts[1]
    elif SOURCE == "dbimages_fk":
        # strip fk and p
        image_id = name[2:].rstrip("p")
    else:
        image_id = name.rstrip("p")

    image_ids.append(image_id)

    # gets matching plantlist filename
    if SOURCE == "warp":
        plant_file = SEQ_DIR / f"{image_id}p{CCD}-{REF_ID}p{CCD}.plantList"
        if not plant_file.exists():
            sys.exit("Missing plantlist")
    else:
        # check ccd subfolder then image_id root
        plant_file = find_plantlist(image_id, CCD)

    # use the frame PSF for dbimages, or the global PSF_FILE for warps
    if SOURCE == "warp":
        frame_psf = Path(PSF_FILE) if PSF_FILE else None
        if frame_psf is None and MODE == "fake":
            sys.exit("MODE=fake with warp source needs a PSF file.")
    else:
        frame_psf = find_psf_file(image_id, CCD)
        if frame_psf is None and MODE == "fake":
            sys.exit("Can't find, required when MODE= fake.")

    with fits.open(fits_file) as hdul:

        if SOURCE == "warp":
            # warps store the image in hdul[1]; header in hdul[0]
            hdu_used = 1
            image = hdul[hdu_used].data.astype(float)
            header = hdul[0].header
        else:
            # HDU index = CCD + 1 
            hdu_used = CCD + 1
            image = hdul[hdu_used].data.astype(float)
            header = hdul[hdu_used].header

        EXPTIME   = float(header.get("EXPTIME"))
        zeropoint = float(header.get("PHOTZP"))
        gain      = float(header.get("GAIN"))

        # cent_time = MJD-OBS + exptime/2./(3600.*24.)
        if "MJD-OBS" in header:
            # read modified julian date
            mjd = float(header["MJD-OBS"])
            cent_time = mjd + EXPTIME / (2.0 * 86400.0)

        else:
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
    cent_times.append(cent_time)
    plant_tables.append(df)
    exptimes.append(EXPTIME)
    psf_files.append(frame_psf)
    zeropoints.append(zeropoint)
    gains.append(gain)

    
    # print(f"Image shape = {image.shape}")
    # print(f"MJD = {mjd}, cent_time = {cent_time}")
    # print(f"EXPTIME = {EXPTIME}, PHOTZP = {zeropoint}, GAIN = {gain}")
    # print(f"Objects = {len(df)}")


    
    
    
    
    
############# Find common objects    
    
    
    

# Find object IDs that appear in every frame
first_table    = plant_tables[0]
common_objects = []

# loop through every object ID in the first plant tzable
for obj in first_table["index"]:
    found_in_all = True
    # check other frames and see if object exists
    for df in plant_tables[1:]:
        if obj not in df["index"].values:
            found_in_all = False

    if found_in_all:
        common_objects.append(obj)

print(f"\nObjects common to all frames: {len(common_objects)}")

if len(common_objects) == 0:
    sys.exit("No common objects found across all frames.")


# Choose possible objects
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
    sys.exit("nope!")

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










########## Cutout function

def get_cutout(image, x, y, half_size):

    height, width = image.shape

    cx = int(round(x))
    cy = int(round(y))

    x0 = cx - half_size
    x1 = cx + half_size

    y0 = cy - half_size
    y1 = cy + half_size
    
    # boundary check
    if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
        return None

    cutout = image[y0:y1, x0:x1].copy()
    cutout[np.isnan(cutout)] = np.nan
    return cutout




###########fake injection function

def inject_fake(image, x, y, rate, angle, mag, exptime, zeropoint, gain, psf_file):

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

    # plant at unit amplitude to get normalized PSF shape
    fake = mpsf.plant(
        np.array([x]),
        np.array([y]),
        np.array([1.0]),
        image * 0.0,
        useLinePSF=True,
        returnModel=True,
        gain=1.0,
        addNoise=False,
        verbose=False
    )

    # normalize, scale by magnitude, and add realistic Poisson-like noise
    norm_flux = np.sum(fake)

    # expected flux from zeropoint and magnitude
    expected_flux = 10.0 ** ((zeropoint - mag) / 2.5)
    fake = fake * (expected_flux / norm_flux)
    fake += np.random.normal(0., np.sqrt(np.abs(fake) / gain))

    new_image += fake

    return new_image









###### Track movement


# real--- object moves relative to this fixed centre each frame
# fake--- fake is injected at predicted position; cutout still fixed here

first_plant_row = plant_tables[0][
    plant_tables[0]["index"] == obj_index
].iloc[0]

x_centre = first_plant_row["x"]
y_centre = first_plant_row["y"]

print(f"\nReference centre: ({x_centre:.2f}, {y_centre:.2f})")


#get positions and cutouts

cutouts         = []
positions       = []
plant_positions = []

rate_ra  = chosen["rate_ra"]
rate_dec = chosen["rate_dec"]

# use cent_times for dt
cent_time0 = cent_times[0]

print("\nTensor cutouts:")
print("frame   dt_hr    x_used   y_used   "
    "dx_used  dy_used   dx_exp   dy_exp   dx_plant dy_plant"
)

# loop through each frame
for i in range(len(images)):

    image = images[i]
    df    = plant_tables[i]
    mjd   = mjds[i]

    # Find same object in the current frame;s plantlist
    plant_row = df[df["index"] == obj_index].iloc[0]
    x_plant   = plant_row["x"]
    y_plant   = plant_row["y"]

    # Compute elapsed time
    dt_hours = (cent_times[i] - cent_time0) * HOURS_PER_DAY

    dx_expected = (rate_ra  * dt_hours / PIXEL_SCALE)
    dy_expected = -(rate_dec * dt_hours / PIXEL_SCALE)

    # use position from plantlist
    if MODE == "real":

        x = x_plant
        y = y_plant

    # predict where object should be and inject fake
    elif MODE == "fake":

        # first frame centre plus motion plus offset
        x = x_centre + dx_expected + X_OFFSET
        y = y_centre + dy_expected + Y_OFFSET

        image = inject_fake(
            image, x, y,
            chosen["rate"], chosen["angle"], chosen["mag"],
            exptimes[i], zeropoints[i], gains[i], psf_files[i]
        )

    else:
        sys.exit("errorr")

    # always extract cutout centreed on the fixed first frame reference
    cutout = get_cutout(image, x_centre, y_centre, HALF_SIZE)
    cutouts.append(cutout)
    positions.append((x, y))
    plant_positions.append((x_plant, y_plant))

    dx_used  = x - x_centre
    dy_used  = y - y_centre
    dx_plant = x_plant - x_centre
    dy_plant = y_plant - y_centre

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



# just to fix naming stuff and avoid overrides
if SOURCE == "warp":
    run_type = "warp_DIFFEXP"
elif SOURCE == "dbimages_fk":
    run_type = "dbimages_fk"
else:
    run_type = "dbimages_nonfk"

first_id = image_ids[0] 
last_id = image_ids[-1]

run_label = (
    f"{run_type}_CCD{CCD}_REF{REF_ID}_"
    f"{first_id}_to_{last_id}_N{len(images)}_{MODE}"
)

# Save tensor

out_npz = OUT_DIR / f"tensor_{run_label}_OBJ{obj_index}.npz"


np.savez_compressed(
    out_npz,
    tensor = tensor,
    positions = np.array(positions),
    plant_positions = np.array(plant_positions),
    mjds = np.array(mjds),
    cent_times = np.array(cent_times),
    obj_index = obj_index,
    mag = chosen["mag"],
    rate = chosen["rate"],
    rate_ra = chosen["rate_ra"],
    rate_dec = chosen["rate_dec"],
)

print()
print(f"Saved tensor: {out_npz}")


# plot tensor frames


n = tensor.shape[0]

cols = 5
rows = int(np.ceil(n / cols))
vmin = np.nanpercentile(tensor, 1)
vmax = np.nanpercentile(tensor, 99)

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

        # red cross = fixed reference at cutout centre
        ax.plot(HALF_SIZE, HALF_SIZE, "+", color="red", markersize=12)

        # cyan dot = object's actual position relative to the reference centre
        #  shows the motion drifting across frames
        x_used, y_used = positions[i]
        # need to convert first so it fits cuotutout
        x_rel = HALF_SIZE + (x_used - x_centre)
        y_rel = HALF_SIZE + (y_used - y_centre)
        ax.plot(x_rel, y_rel, ".", color="cyan", markersize=6)

        xp, yp = plant_positions[i]
        ax.set_title(
            f"frame {i}\nused=({x_used:.1f},{y_used:.1f})\nplant=({xp:.1f},{yp:.1f})",
            fontsize=7
        )

    ax.axis("off")

plt.suptitle(
    f"{run_type} | mode={MODE} | CCD={CCD} | REF/anchor={REF_ID}\n"
    f"IDs {first_id} to {last_id} | N={tensor.shape[0]} | "
    f"Object {obj_index} | mag={chosen['mag']:.2f} | shape={tensor.shape}\n"
    f"red=reference centre  cyan=object position"
)

plt.tight_layout()

out_png = OUT_DIR / f"tensor_{run_label}_OBJ{obj_index}.png"

plt.savefig(out_png, dpi=150)
plt.show()

print(f"Saved plot: {out_png}")
