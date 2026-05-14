"""
Open a DIFFEXP or fk image, read the matching plantList, choose a random/most mag fake KBO,
cut out a small image around it, and display/save the result.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from pathlib import Path
import sys


FITS_FILE = input("Enter FITS file: ")
PLANT_FILE = input("Enter plant file: ")
HALF_SIZE = input("Enter cutout size (def=50): ")

if HALF_SIZE == "":
    HALF_SIZE = 50
else:
    HALF_SIZE = int(HALF_SIZE)

if FITS_FILE == "":
    FITS_FILE = "/arc/projects/classy/warps/2022-08-01-AS1_July/15/DIFFEXP-2773082-2773118-15.fits"

if PLANT_FILE == "":
    PLANT_FILE = "/arc/projects/classy/warps/2022-08-01-AS1_July/15/2773082p15-2773118p15.plantList"




#########
# open the fits file and read the image and header
fits_path = Path(FITS_FILE)
if not fits_path.exists():
    sys.exit("FITS file not found.")

with fits.open(fits_path) as hdul:
    hdul.info()
    print()

    # DIFFEXP files have image data in HDU 1
    # fk dbimages have image data in HDU 0
    #  loop fiinds the first HDU that actually contains image datas
    image = None
    header = None
    hdu_used = None

    for i, hdu in enumerate(hdul):
        if hdu.data is not None:
            image = hdu.data
            header = hdu.header
            hdu_used = i
            break

    if image is None:
        sys.exit("No image data found")

height, width = image.shape
print(f"Image shape: {image.shape}")


# read the plant list
plant_path = Path(PLANT_FILE)
if not plant_path.exists():
    sys.exit("Plant list not found")

col_names = [
    'index', 'ra', 'dec', 'x', 'y',
    'rate', 'angle', 'rate_ra', 'rate_dec',
    'mag', 'psf_amp', 'g_i'
]

df = pd.read_csv(
    plant_path,
    sep=r'\s+',
    comment='#',
    names=col_names,
    dtype=float,
)

print(f"Planted objects total: {len(df)}")


# filter only objects that allow a full cutout
H = HALF_SIZE

# cutout of size 2*H x 2*H must fit inside the image,
# so the centre must be at least H pixels from each edge
safe = df[
    (df['x'] >= H) & (df['x'] < width  - H) &
    (df['y'] >= H) & (df['y'] < height - H)
].copy()

print(f"Objects inside border (half-size={H}): {len(safe)}")

if safe.empty:
    sys.exit("No objects are inside the image")


# choose the brightest/low mag object
safe_sorted = safe.sort_values('mag')          # ascending = brightest first
row = safe_sorted.iloc[0]

# choose a random object
#row = safe.sample(n=1).iloc[0]


# print selected object info
obj_index = int(row['index'])
obj_x     = row['x']
obj_y     = row['y']
obj_mag   = row['mag']
obj_rate  = row['rate']
obj_angle = row['angle']
obj_rra   = row['rate_ra']
obj_rdec  = row['rate_dec']
obj_amp   = row['psf_amp']

print("\nSelected object:")
print(f"index = {obj_index}")
print(f"x, y = {obj_x:.2f}, {obj_y:.2f}")
print(f"mag = {obj_mag:.2f}")
print(f"rate = {obj_rate:.2f} arcsec/hr, angle = {obj_angle:.2f} deg")
print(f"rate_ra = {obj_rra:.2f}  rate_dec = {obj_rdec:.2f} arcsec/hr")
print(f"psf_amp = {obj_amp:.5f}\n")


# image[y_row, x_col]
# integer pixel at the object centre
cx = int(round(obj_x))   # column x
cy = int(round(obj_y))   # row y

# extract cutout
cutout = image[
    cy - H : cy + H,
    cx - H : cx + H
].copy()

print(f"Cutout shape: {cutout.shape} centred at pixel ({cx}, {cy})")


# display cutout, take only valid pixels
finite_cut = cutout[np.isfinite(cutout)]
if finite_cut.size == 0:
    sys.exit("The cutout is NaN.")


# grayscale mapping
vmin = np.percentile(finite_cut, 1)
vmax = np.percentile(finite_cut, 99)


# PLOTTT
fig, ax = plt.subplots(figsize=(6, 6))

im = ax.imshow(
    cutout,
    origin='lower',
    cmap='gray',
    vmin=vmin,
    vmax=vmax,
)

plt.colorbar(im, ax=ax, label='Pixel Value (ADU)')


# mark where the fake object is supposed to be
ax.plot(
    H, H,
    '+',
    color='red',
    markersize=14,
    markeredgewidth=2,
    label=f'Planted Obj (mag={obj_mag:.2f})'
)

ax.set_xlabel('x')
ax.set_ylabel('y')


# just getting the file IDs and CCD from the filename for the title
name = Path(FITS_FILE).stem

if name.startswith("DIFFEXP"):
    # DIFFEXP-2773082-2773118-15.fits
    parts = name.split("-")

    image_type = "DIFFEXP"
    image_id = parts[1]
    ref_id = parts[2]
    ccd_id = parts[3]

    # title with info from file
    title = (
        f"DIFFEXP {image_id}-{ref_id} | CCD {ccd_id} | Object {obj_index}\n"
        f"x={obj_x:.1f}  y={obj_y:.1f}  "
        f"Rate={obj_rate:.2f}\"/hr  Angle={obj_angle:.1f}°"
    )

    outfile = f"DIFFEXP_{image_id}_{ref_id}_CCD{ccd_id}_OBJ{obj_index}.png"

elif name.startswith("fk"):
    # fk2773082p15.fits
    fk_info = name.replace("fk", "")
    image_id, ccd_id = fk_info.split("p")

    # title with info from file
    title = (
        f"FK {image_id} | CCD {ccd_id} | Object {obj_index}\n"
        f"x={obj_x:.1f}  y={obj_y:.1f}  "
        f"Rate={obj_rate:.2f}\"/hr  Angle={obj_angle:.1f}°"
    )

    outfile = f"FK_{image_id}_CCD{ccd_id}_OBJ{obj_index}.png"


ax.set_title(title)
ax.legend(loc='upper right', fontsize=8)

plt.tight_layout()

plt.savefig(outfile, dpi=150)
plt.show()

print(f"Saved: {outfile}")