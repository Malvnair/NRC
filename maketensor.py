"""
maketensor5.py

Takes one CCD sequence,
finds brightest planted object across frames,
extracts 200 x 200 cutouts,
and saves them as a tensor.

Supports warp, dbimages_fk, and dbimages sources.
"""

import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from pathlib import Path

sys.path.append('/arc/home/malvnair/trippy')
from trippy import psf




########## config and arg stuff


def load_json_config(path):
    """Load a JSON run config and return it as a dict."""
    with open(path) as f:
        return json.load(f)


def parse_args():
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description="Build a tensor cutout from a CLASSY warp or dbimages sequence."
    )

    parser.add_argument("--visit",    type=str,   help="Visit name, e.g. 2022-08-01-AS1_July")
    parser.add_argument("--ccd",      type=int,   help="CCD number")
    parser.add_argument("--ref-id",   type=str,   help="Reference image ID")
    parser.add_argument("--mode",     type=str,   choices=["real", "fake"], help="real or fake")
    parser.add_argument("--n-frames", type=int,   help="Max number of frames to use")
    parser.add_argument("--half-size",type=int,   help="Cutout half-size in pixels")
    parser.add_argument("--psf-file", type=str,   default=None, help="PSF file (fake mode)")
    parser.add_argument("--config",   type=str,   default=None, help="Path to JSON run config")
    parser.add_argument("--out-dir",  type=str,   default="./tensors", help="Output directory")
    parser.add_argument("--keep-template-order", action="store_true", help="Use visit-list order instead of chronological order")
    parser.add_argument("--source",   type=str,   choices=["warp", "dbimages_fk", "dbimages"],
                        help="Source type: warp, dbimages_fk, or dbimages")
    args = parser.parse_args()

    # start with default values
    cfg = {
        "visit":     None,
        "ccd":       15,
        "ref_id":    None,
        "mode":      "real",
        "n_frames":  8,
        "half_size": 100,
        "psf_file":  None,
        "out_dir":      "./tensors",
        "sort_by_time": True,
        "source":    "warp",
        "image_ids": None,
    }

    # use json values if given
    if args.config:
        json_cfg = load_json_config(args.config)
        for key in cfg:
            json_key = key.replace("_", "-")
            if json_key in json_cfg:
                cfg[key] = json_cfg[json_key]
            elif key in json_cfg:
                cfg[key] = json_cfg[key]

    # command line values win last
    if args.visit     is not None: cfg["visit"]     = args.visit
    if args.ccd       is not None: cfg["ccd"]       = args.ccd
    if args.ref_id    is not None: cfg["ref_id"]    = args.ref_id
    if args.mode      is not None: cfg["mode"]      = args.mode
    if args.n_frames  is not None: cfg["n_frames"]  = args.n_frames
    if args.half_size is not None: cfg["half_size"] = args.half_size
    if args.psf_file  is not None: cfg["psf_file"]  = args.psf_file
    if args.out_dir   is not None: cfg["out_dir"]   = args.out_dir
    if args.source    is not None: cfg["source"]    = args.source
    if args.keep_template_order:   cfg["sort_by_time"] = False

    # visit is only required for warp mode
    if cfg["source"] == "warp" and cfg["visit"] is None:
        parser.error("--visit is required for warp source (or set 'visit' in your JSON config).")

    # ref_id is required for dbimages unless an explicit image_ids list is given
    if cfg["source"] in ("dbimages_fk", "dbimages") and cfg["ref_id"] is None and cfg["image_ids"] is None:
        parser.error("--ref-id is required for dbimages source unless 'image_ids' is set in your JSON config.")

    return cfg




########## settings


# pixel offsets applied to fake injection position

X_OFFSET = 5.0
Y_OFFSET = 5.0

HOURS_PER_DAY = 24.0

PLANT_COLS = [
    "index", "ra", "dec", "x", "y",
    "rate", "angle", "rate_ra", "rate_dec",
    "mag", "psf_amp", "g_i"
]

# paths to classy data
VISIT_LIST_ROOT = Path("/arc/projects/classy/visitLists")
WARP_ROOT       = Path("/arc/projects/classy/warps")

# dbimages path for later
DB_ROOT = Path("/arc/projects/classy/dbimages")




########## dbimages helper functions


# find plantlist in either the ccd subfolder or the imageid folder
def find_plantlist(image_id, ccd):
    path_ccd = DB_ROOT / image_id / f"ccd{ccd}" / f"{image_id}p{ccd}.plantList"
    path_top = DB_ROOT / image_id / f"{image_id}p{ccd}.plantList"
    if path_ccd.exists():
        return path_ccd
    if path_top.exists():
        return path_top
    sys.exit(f"plantList not found for image_id={image_id} ccd={ccd}.")


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




########## warp visit class


class WarpVisit:
    """
    Find the teplate visit list, read the ordrered image ids from it,
    find matchin fits and plantlist files, return the file lists
    """

    def __init__(self, visit_name, ccd, ref_id, n_frames):
        self.visit_name = visit_name
        self.ccd        = ccd
        self.ref_id     = str(ref_id)
        self.n_frames   = n_frames

        # filled after setup runs
        self.visit_list_path = None
        self.warp_dir        = None
        self.image_ids       = []
        self.fits_files      = []
        self.plant_files     = []

    def setup(self):
        """
        Fix paths and build the ordered file lists
        """

        # template visit list path
        self.visit_list_path = (
            VISIT_LIST_ROOT
            / self.visit_name
            / f"{self.visit_name}_visit_list.txt"
        )

        if not self.visit_list_path.exists():
            sys.exit(f"Visit list not found: {self.visit_list_path}")

        # warp folder for this visit and ccd
        self.warp_dir = WARP_ROOT / self.visit_name / str(self.ccd)

        if not self.warp_dir.is_dir():
            sys.exit(f"Warp directory not found: {self.warp_dir}")

        # read image ids from the visit list
        template_ids = []
        with open(self.visit_list_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    template_ids.append(line)

        if len(template_ids) == 0:
            sys.exit("Template visit list is empty.")

        # only keep files from the visit list
        # keep the template order
        fits_files  = []
        plant_files = []
        used_ids    = []

        for image_id in template_ids:

            fits_matches = sorted(
                self.warp_dir.glob(f"DIFFEXP-{image_id}-*-{self.ccd}.fits")
            )

            plant_matches = sorted(
                self.warp_dir.glob(f"{image_id}p{self.ccd}-*p{self.ccd}.plantList")
            )

            if len(fits_matches) == 1 and len(plant_matches) == 1:
                fits_files.append(fits_matches[0])
                plant_files.append(plant_matches[0])
                used_ids.append(image_id)

            elif len(fits_matches) == 0 or len(plant_matches) == 0:
                print(f"Skipping.")

            else:
                print(f"Skipping.")

        if len(fits_files) == 0:
            sys.exit("No matching files found in the warp directory.")

        # keep only the number of frames requested
        self.fits_files = fits_files[:self.n_frames]
        self.plant_files = plant_files[:self.n_frames]
        self.image_ids = used_ids[:self.n_frames]

        return self




########## dbimages visit class


class DbImageVisit:

    def __init__(self, source, ccd, ref_id, n_frames, image_ids=None):
        self.source    = source      # "dbimages_fk" or "dbimages"
        self.ccd       = ccd
        self.ref_id    = str(ref_id)
        self.n_frames  = n_frames
        # if image_ids provided directly, use them, otherwise discover from refid
        self._given_ids = image_ids

        # filled after setup runs
        self.image_ids   = []
        self.fits_files  = []
        self.plant_files = []
        self.psf_files   = []

    def setup(self):
        """
        Get image idss, build fits/plant/psf file lists.
        Returns self for chaining.
        """

        # use explicit list if given, otherwise walk DB_ROOT from ref_id
        if self._given_ids is not None:
            all_ids = [str(x) for x in self._given_ids]
        else:
            all_ids = discover_sequence(self.ref_id)

        if len(all_ids) == 0:
            sys.exit(f"No found")

        # build fits file list depending on fk or non-fk
        fits_files  = []
        plant_files = []
        psf_files   = []
        used_ids    = []

        for image_id in all_ids:

            if self.source == "dbimages_fk":
                # mosaic fk file, no ccd subfolder so CCD selected by HDU
                fits_path = DB_ROOT / image_id / f"fk{image_id}p.fits"
            else:
                # mosaic non-fk file, no ccd subfolder CCD selected by HDU
                fits_path = DB_ROOT / image_id / f"{image_id}p.fits"

            if not fits_path.exists():
                print(f"Skipping.")
                continue

            plant_path = find_plantlist(image_id, self.ccd)
            psf_path   = find_psf_file(image_id, self.ccd)

            fits_files.append(fits_path)
            plant_files.append(plant_path)
            psf_files.append(psf_path)
            used_ids.append(image_id)

        if len(fits_files) == 0:
            sys.exit("None found.")

        # keep only the number of frames requested
        self.fits_files  = fits_files[:self.n_frames]
        self.plant_files = plant_files[:self.n_frames]
        self.psf_files   = psf_files[:self.n_frames]
        self.image_ids   = used_ids[:self.n_frames]

        return self




########## wcs and pixel scale stuff


def get_pixel_scale_from_wcs(wcs_obj):
    """
    Get pixel scale in arcsec/pixel from a WCS object.
    """
    try:
        # astropy gives scale in degrees
        scales = wcs_obj.proj_plane_pixel_scales()
        # convert to arcsec per pixel
        pixel_scale = float(np.mean([s.value for s in scales]) * 3600.0)
        return pixel_scale
    except Exception:
        pass

def find_usable_wcs(hdul, df_plant):
    """
    Find a WCS that works for this file.
    Tries the image HDU header first, then the primary header.
    """

    # find the first HDU with 2D image data
    image_hdu_index = None
    for k, hdu in enumerate(hdul):
        if hdu.data is not None and hdu.data.ndim == 2:
            image_hdu_index = k
            break

    if image_hdu_index is None:
        sys.exit("Not found.")

    # try the image HDU header first
    for label, header in [
        ("image_hdu_header", hdul[image_hdu_index].header),
        ("primary_header",   hdul[0].header),
    ]:
        wcs_obj = WCS(header, naxis=2)
        px, py  = wcs_obj.all_world2pix(
            df_plant["ra"].values[:5], df_plant["dec"].values[:5], 0
        )
        if np.all(np.isfinite(px)) and np.all(np.isfinite(py)):
            return wcs_obj, image_hdu_index, label

    sys.exit("Could not find a usable WCS")


def find_usable_wcs_dbimages(hdu_header, df_plant):
    """
    Build a WCS from a dbimages CCD HDU header.
    """

    wcs_obj = WCS(hdu_header, naxis=2)
    px, py  = wcs_obj.all_world2pix(
        df_plant["ra"].values[:5], df_plant["dec"].values[:5], 0
    )

    if not (np.all(np.isfinite(px)) and np.all(np.isfinite(py))):
        sys.exit("Could not find.")

    return wcs_obj, "dbimages_ccd_hdu_header"


########## object selection stuff


def find_common_objects(plant_tables):
    """
    Find object ids that showw in every frame's plantList.
    Returns a list of integer IDs.
    """
    first_table = plant_tables[0]
    common_objects = []

    # loop through every object ID in the first plant table
    for obj in first_table["index"]:
        found_in_all = True
        # check other frames and see if object exists
        for df in plant_tables[1:]:
            if obj not in df["index"].values:
                found_in_all = False

        if found_in_all:
            common_objects.append(obj)

    return common_objects


def filter_safe_objects(common_objects, plant_tables, wcs_list, image_shapes, half_size):
    """
    Filter common objects to those safely away from edges in every frame.
    Uses WCS-derived pixel positions instead of plantList x,y.
    """
    safe_indices = set(common_objects)

    for df, wcs_obj, shape in zip(plant_tables, wcs_list, image_shapes):
        height, width = shape

        # get wcs pixel positions for this frame
        df = df.copy()
        px, py = wcs_obj.all_world2pix(df["ra"].values, df["dec"].values, 0)
        df["wcs_x"] = px
        df["wcs_y"] = py

        safe_here = df[
            (df["wcs_x"] >= half_size) &
            (df["wcs_x"] <  width  - half_size) &
            (df["wcs_y"] >= half_size) &
            (df["wcs_y"] <  height - half_size)
        ]

        safe_indices = safe_indices.intersection(set(safe_here["index"].values))

    return safe_indices




########## cutout and fake injection stuff


def get_cutout(image, x, y, half_size):
    """Extract a fixed-size cutout on (x, y)."""

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


def inject_fake(image, x, y, rate, angle, mag, exptime,
                zeropoint, gain, psf_file, pixel_scale):
    """Inject a trailed fake source into a copy of image."""

    new_image = image.copy().astype(float)

    mpsf = psf.modelPSF(restore=str(psf_file))

    # trailed
    mpsf.line(
        rate,
        angle,
        exptime / 3600.0,
        pixScale=pixel_scale,
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
    
    plt.imshow(fake, origin = "lower")
    plt.colorbar()
    
    cfg = parse_args()

    OUT_DIR  = Path(cfg["out_dir"])
    OUT_DIR.mkdir(exist_ok=True)
    test = OUT_DIR / f"test.png"

    plt.savefig(test)
    new_image += fake

    return new_image




########## main part


def main():

    cfg = parse_args()

    VISIT = cfg["visit"]
    CCD = cfg["ccd"]
    REF_ID = cfg["ref_id"]
    MODE = cfg["mode"]
    N_FRAMES = cfg["n_frames"]
    HALF_SIZE = cfg["half_size"]
    PSF_FILE = cfg["psf_file"]
    OUT_DIR = Path(cfg["out_dir"])
    SORT_BY_TIME = cfg["sort_by_time"]
    SOURCE = cfg["source"]
    IMAGE_IDS_CFG = cfg.get("image_ids", None)
    OUT_DIR.mkdir(exist_ok=True)


    ########## build up fits sequence to use!

    if SOURCE == "warp":

        visit = WarpVisit(VISIT, CCD, REF_ID, N_FRAMES).setup()

        fits_files   = visit.fits_files
        plant_files  = visit.plant_files
        IMAGE_IDS    = visit.image_ids

        print(f"\nVisit: {VISIT}")
        print(f"CCD: {CCD}")
        print(f"REF_ID: {REF_ID}")
        print(f"Mode:{MODE}")
        print(f"Frames: {len(fits_files)}")
        print(f"Visit list: {visit.visit_list_path}")
        print(f"Warp dir:{visit.warp_dir}")

    elif SOURCE in ("dbimages_fk", "dbimages"):

        visit = DbImageVisit(
            SOURCE, CCD, REF_ID, N_FRAMES, image_ids=IMAGE_IDS_CFG
        ).setup()

        fits_files   = visit.fits_files
        plant_files  = visit.plant_files
        IMAGE_IDS    = visit.image_ids

        print(f"\nSource: {SOURCE}")
        print(f"CCD: {CCD}")
        print(f"REF_ID: {REF_ID}")
        print(f"Mode: {MODE}")
        print(f"Frames: {len(fits_files)}")

    else:
        sys.exit(f"failll...qwq.")

    # make sure its consecutive
    print("\nFITS files to use:")
    for f in fits_files:
        print(f"  {f.name}")


        
######## Get data from the fits files


    # initialize empty lists to load images and plantlists
    images       = []
    mjds         = []
    cent_times   = []
    plant_tables = []
    image_ids    = []
    exptimes     = []
    psf_files    = []
    zeropoints   = []
    gains        = []
    wcs_list     = []
    wcs_sources  = []
    pixel_scales = []

    for i_frame, (fits_file, plant_file) in enumerate(zip(fits_files, plant_files)):

        image_id = IMAGE_IDS[i_frame]
        image_ids.append(image_id)

        # warp uses global psf file; dbimages uses per-frame lookup
        if SOURCE == "warp":
            frame_psf = Path(PSF_FILE) if PSF_FILE else None
        else:
            frame_psf = visit.psf_files[i_frame]
            if frame_psf is None and MODE == "fake":
                sys.exit(f"No PSF found.")

        df = pd.read_csv(
            plant_file,
            sep=r"\s+",
            comment="#",
            names=PLANT_COLS,
            dtype=float
        )

        # convert ids so they not decimals
        df["index"] = df["index"].astype(int)

        with fits.open(fits_file) as hdul:

            if SOURCE == "warp":
                wcs_obj, img_hdu_idx, wcs_src = find_usable_wcs(hdul, df)
                image  = hdul[img_hdu_idx].data.astype(float)
                header = hdul[0].header
            else:
                # HDU index = CCD + 1
                hdu_used = CCD + 1
                image  = hdul[hdu_used].data.astype(float)
                header = hdul[hdu_used].header
                #header['RADESYSa']='FK5 ' 
                wcs_obj, wcs_src = find_usable_wcs_dbimages(header, df)

            EXPTIME   = float(header.get("EXPTIME") or 0.0)
            zeropoint = float(header.get("PHOTZP")  or 0.0)
            gain      = float(header.get("GAIN")    or 3.0)

            if "MJD-OBS" in header:
                mjd = float(header["MJD-OBS"])
            else:
                print(f"No MJD-OBS found for {fits_file.name}, using 0.0")
                mjd = 0.0
            cent_time = mjd + EXPTIME / (2.0 * 86400.0)

        pixel_scale = get_pixel_scale_from_wcs(wcs_obj)

        images.append(image)
        mjds.append(mjd)
        cent_times.append(cent_time)
        plant_tables.append(df)
        exptimes.append(EXPTIME)
        psf_files.append(frame_psf)
        zeropoints.append(zeropoint)
        gains.append(gain)
        wcs_list.append(wcs_obj)
        wcs_sources.append(wcs_src)
        pixel_scales.append(pixel_scale)

        # print(f"Image shape = {image.shape}")
        # print(f"MJD = {mjd}, cent_time = {cent_time}")
        # print(f"EXPTIME = {EXPTIME}, PHOTZP = {zeropoint}, GAIN = {gain}")
        # print(f"Objects = {len(df)}")
        # print(f"WCS source = {wcs_src}, pixel_scale = {pixel_scale:.4f} arcsec/pix")



    ######## sort frames if needed

    # visit list  time ordered

    if SORT_BY_TIME:
        sort_order = np.argsort(cent_times)
        images       = [images[k]       for k in sort_order]
        mjds         = [mjds[k]         for k in sort_order]
        cent_times   = [cent_times[k]   for k in sort_order]
        plant_tables = [plant_tables[k] for k in sort_order]
        exptimes     = [exptimes[k]     for k in sort_order]
        psf_files    = [psf_files[k]    for k in sort_order]
        zeropoints   = [zeropoints[k]   for k in sort_order]
        gains        = [gains[k]        for k in sort_order]
        wcs_list     = [wcs_list[k]     for k in sort_order]
        wcs_sources  = [wcs_sources[k]  for k in sort_order]
        pixel_scales = [pixel_scales[k] for k in sort_order]
        image_ids    = [image_ids[k]    for k in sort_order]
        fits_files   = [fits_files[k]   for k in sort_order]
        plant_files  = [plant_files[k]  for k in sort_order]

    ############# Find common objects

    common_objects = find_common_objects(plant_tables)

    print(f"\nObjects common to all frames: {len(common_objects)}")

    if len(common_objects) == 0:
        sys.exit("No common objects found across all frames.")

    # Choose possible objects
    first_table = plant_tables[0]
    possible = first_table[
        first_table["index"].isin(common_objects)
    ].copy()

    # use WCS positions for edge filterring
    image_shapes = [img.shape for img in images]
    safe_indices = filter_safe_objects(
        common_objects, plant_tables, wcs_list, image_shapes, HALF_SIZE
    )

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


    ###### Track movement

    # real--- object moves relative to this fixed centre each frame
    # fake--- fake is injected at predicted position; cutout still fixed here

    # first frame gives the reference centre
    first_plant_row = plant_tables[0][
        plant_tables[0]["index"] == obj_index
    ].iloc[0]

    ref_ra  = first_plant_row["ra"]
    ref_dec = first_plant_row["dec"]

    # WCS centre from the first frame
    wcs0 = wcs_list[0]
    x_centre_arr, y_centre_arr = wcs0.all_world2pix([ref_ra], [ref_dec], 0)
    x_centre = float(x_centre_arr[0])
    y_centre = float(y_centre_arr[0])

    print(f"\nReference centre: ({x_centre:.2f}, {y_centre:.2f})")


    ############get positions and cutouts

    cutouts         = []
    positions       = []
    plant_positions = []
    used_image_ids  = []      # image IDs for frames that made it into the tensor

    rate_ra  = chosen["rate_ra"]
    rate_dec = chosen["rate_dec"]

    # use cent_times for dt
    cent_time0 = cent_times[0]

    skipped_frames = []

    print("\nTensor cutouts:")
    print("frame   dt_hr    x_used   y_used   "
          "dx_used  dy_used   dx_exp   dy_exp   dx_plant dy_plant"
    )

    # loop through each frame
    for i in range(len(images)):

        image = images[i]
        df    = plant_tables[i]
        mjd   = mjds[i]
        wcs_i = wcs_list[i]
        pscale = pixel_scales[i]

        # Find same object in the current frame;s plantlist
        plant_row = df[df["index"] == obj_index].iloc[0]

        # WCS position for this object in this frame
        px_arr, py_arr = wcs_i.all_world2pix(
            [plant_row["ra"]], [plant_row["dec"]], 0
        )
        x_plant = float(px_arr[0])
        y_plant = float(py_arr[0])

        # Compute elapsed time
        dt_hours = (cent_times[i] - cent_time0) * HOURS_PER_DAY

        dx_expected = (rate_ra  * dt_hours / pscale)
        dy_expected = -(rate_dec * dt_hours / pscale)

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
                exptimes[i], zeropoints[i], gains[i], psf_files[i],
                pscale
            )

        else:
            sys.exit("errorr")

        # always extract cutout centreed on the fixed first frame reference
        cutout = get_cutout(image, x_centre, y_centre, HALF_SIZE)

        cutouts.append(cutout)
        positions.append((x, y))
        plant_positions.append((x_plant, y_plant))
        used_image_ids.append(image_ids[i])

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

    first_id = used_image_ids[0]
    last_id  = used_image_ids[-1]

    run_label = (
        f"{run_type}_CCD{CCD}_REF{REF_ID}_"
        f"{first_id}_to_{last_id}_N{tensor.shape[0]}_{MODE}"
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



    
    
    ##########plot tensor frames

    # one frame per row so it is easier to read

    n = tensor.shape[0]

    # was thinking about two columns later
    # for now just use one column
    panel_w = 4          # inches per panel
    panel_h = 4          # inches per panel
    vmin = np.nanpercentile(tensor, 1)
    vmax = np.nanpercentile(tensor, 99)

    fig, axes = plt.subplots(n, 1, figsize=(panel_w, panel_h * n))

    # fix n=1 plot case
    if n == 1:
        axes = [axes]

    for i in range(n):

        ax = axes[i]

        ax.imshow(
            tensor[i],
            origin="lower",
            cmap="gray",
            vmin=vmin,
            vmax=vmax
        )

        # red cross = fixed reference at cutout centre
        ax.plot(HALF_SIZE, HALF_SIZE, "+", color="purple", markersize=6)

        # cyan dot = object's actual position relative to the reference centre
        #  shows the motion drifting across frames
        x_used, y_used = positions[i]
        x_original, y_original = positions[0]
        # need to convert first so it fits cutout
        x_rel = HALF_SIZE + (x_used - x_centre)
        y_rel = HALF_SIZE + (y_used - y_centre)
        marker_x = HALF_SIZE + (x_original - x_centre)
        marker_y = HALF_SIZE + (y_original - y_centre)
        ax.plot(x_rel, y_rel, ".", color="cyan", markersize=6)
        ax.plot(marker_x, marker_y, "+", color="red", markersize=12)



        xp, yp = plant_positions[i]
        dt_hr = (cent_times[i] - cent_times[0]) * HOURS_PER_DAY
        ax.set_title(
            f"frame {i}  id={used_image_ids[i]}  dt={dt_hr:.3f} hr"
            f"  used=({x_used:.1f},{y_used:.1f})"
            f"  plant=({xp:.1f},{yp:.1f})",
            fontsize=7, loc="left"
        )

        ax.axis("off")

    fig.suptitle(
        f"{run_type} | mode={MODE} | CCD={CCD} | REF/anchor={REF_ID}\n"
        f"IDs {first_id} to {last_id} | N={tensor.shape[0]} | "
        f"Object {obj_index} | mag={chosen['mag']:.2f} | shape={tensor.shape}\n"
        f"red=reference centre  cyan=object position",
        fontsize=9,
        y=0.995
    )

    plt.tight_layout(rect=[0, 0, 1, 1])
    out_png = OUT_DIR / f"tensor_{run_label}_OBJ{obj_index}.png"

    plt.savefig(out_png, dpi=150)
    plt.show()

    print(f"Saved plot: {out_png}")


if __name__ == "__main__":
    main()
