"""
maketensor7.py

Negative wells edition.

TNO tensor generator for the CLASSY survey.

Loads a sequence of warp FITS images once, then generates fake moving
TNOs by randomly sampling orbits and sky positions, inject them into real
warp backgrounds, and saving one .npz tensor per sample.
"""

import sys
import json
import argparse
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from pathlib import Path

sys.path.append("/arc/home/malvnair/trippy")
from trippy import psf as trippy_psf


####################
#Paths
##############


VISIT_LIST_ROOT = Path("/arc/projects/classy/visitLists")
WARP_ROOT       = Path("/arc/projects/classy/warps")
DB_ROOT         = Path("/arc/projects/classy/dbimages")

HOURS_PER_DAY   = 24.0
ARCSEC_PER_DEG  = 3600.0






def find_psf_file_from_dbimages(image_id, ccd):
    """
    Look up the TRIPPy PSF file for a given science image ID and CCD number.
    """
    image_id = str(image_id)
    # for the ccd specific psf file
    path_ccd = DB_ROOT / image_id / f"ccd{ccd}" / f"{image_id}p{ccd}.psf.fits"
    path_top = DB_ROOT / image_id / f"{image_id}p{ccd}.psf.fits"
    
    # Use the ccd specific one first
    if path_ccd.exists():
        return path_ccd
    if path_top.exists():
        return path_top
    return None



def project_track_position(wcs_obj, pixel_scale, ra_ref, dec_ref, motion, dt_hr):
    """Calculate where the fake object/residual should be on a fram"""
    x_ref_arr, y_ref_arr = wcs_obj.all_world2pix([ra_ref], [dec_ref], 0)
    x_ref = float(x_ref_arr[0])
    y_ref = float(y_ref_arr[0])
    dx_pix =  motion["rate_ra"]  * dt_hr / pixel_scale
    dy_pix = -motion["rate_dec"] * dt_hr / pixel_scale
    # predicted pixel position
    return x_ref + dx_pix, y_ref + dy_pix


####################
# Config/argument handling
###############


# Open json file and return content as dictionary
def load_json_config(path):
    with open(path) as f:
        return json.load(f)


    
def parse_args():
    """Command args override JSON config."""

    # create arg parser object
    parser = argparse.ArgumentParser(
        description="Synthetic TNO tensor generator for CLASSY warps."
    )
    parser.add_argument("--visit", type=str,  help="Visit name, e.g. 2022-08-01-AS1_July")
    parser.add_argument("--ccd", type=int,  help="CCD number")
    # number of fake tnos
    parser.add_argument("--num-samples", type=int,  help="Number of fake TNOs to generate")
    parser.add_argument("--n-frames", type=int,  help="Max science frames to load")
    parser.add_argument("--half-size", type=int,  help="Cutout half-size in pixels")
    parser.add_argument("--out-dir", type=str,  help="Output directory")
    # JSON config
    parser.add_argument("--config", type=str, default=None, help="JSON run config")
    parser.add_argument("--keep-template-order", action="store_true",
                        help="Use visit-list order instead of MJD order")
    # real image background for debugging
    parser.add_argument("--debug-fake-only", action="store_true",
                        help="Zero the background")
    parser.add_argument("--save-plots", action="store_true",
                        help="Save a PNG")

    args = parser.parse_args()

    # defaults
    cfg = {
        "visit":None,
        "ccd":15,
        "num_samples":1,
        "n_frames":8,
        "half_size":100,
        "out_dir": "./tensors",
        "sort_by_time":True,
        "debug_fake_only": False,
        "save_plots":False,
        # negative wells 
        "add_noise_to_negative_wells": False, 
    }

    # JSON file stuff
    if args.config:
        jcfg = load_json_config(args.config)
        for key in cfg:
            # allow hyphens
            jkey = key.replace("_", "-")
            if jkey in jcfg:
                cfg[key] = jcfg[jkey]
            elif key in jcfg:
                cfg[key] = jcfg[key]

    # CLI layer (override json)
    if args.visit is not None: cfg["visit"] = args.visit
    if args.ccd is not None: cfg["ccd"] = args.ccd
    if args.num_samples is not None: cfg["num_samples"] = args.num_samples
    if args.n_frames is not None: cfg["n_frames"] = args.n_frames
    if args.half_size is not None: cfg["half_size"] = args.half_size
    if args.out_dir is not None: cfg["out_dir"] = args.out_dir
    if args.keep_template_order: cfg["sort_by_time"] = False
    if args.debug_fake_only: cfg["debug_fake_only"] = True
    if args.save_plots: cfg["save_plots"]  = True

    if cfg["visit"] is None:
        parser.error("--visit is required.")

    return cfg


###############
# WarpDataset
# Loads an ordered sequence of science warp FITS files for one visit/CCD.
####################

class WarpDataset:
    """
    Reads the science visit list, finds matching DIFFEXP FITS files, and loads
    everything into memory: images, WCS, zeropoints, gains, MJDs, exptimes.
    """

    # intialize the dataset object with details
    def __init__(self, visit, ccd, n_frames, sort_by_time=True):
        self.visit = visit
        self.ccd = ccd
        self.n_frames = n_frames
        self.sort_by_time = sort_by_time

        self.images = []
        self.mjds = []
        self.cent_times = []
        self.exptimes = []
        self.zeropoints = []
        self.gains = []
        self.wcs_list = []
        self.pixel_scales = []
        self.image_ids = []
        self.fits_files = []
        self.psf_files  = []  


    # loading the datset
    def load(self):
        """Discover files, open FITS, fills all lists."""
        fits_files, image_ids = self._discover_files()
        for fits_file, image_id in zip(fits_files, image_ids):
            self._load_one(fits_file, image_id)
        if self.sort_by_time:
            self._sort_chronological()
        # print summary of loaded frames
        for i, (fid, f) in enumerate(zip(self.image_ids, self.fits_files)):
            psf_display = self.psf_files[i].name if self.psf_files[i] else "NOT FOUND"
            print(f"  [{i}] id={fid}  zp={self.zeropoints[i]:.4f}  "
                  f"exptime={self.exptimes[i]:.1f}s  mjd={self.mjds[i]:.6f}  "
                  f"psf={psf_display}")
        return self

    
    def _discover_files(self):
        """
        Read the visit list to get ordered science image IDs.
        Match each ID to its DIFFEXP file in the warp directory.
        """
        
        # build up the path
        visit_list_path = (
            VISIT_LIST_ROOT / self.visit / f"{self.visit}_visit_list.txt"
        )
        if not visit_list_path.exists():
            sys.exit(f"Visit list not found: {visit_list_path}")

        warp_dir = WARP_ROOT / self.visit / str(self.ccd)
        if not warp_dir.is_dir():
            sys.exit(f"Warp directory not found: {warp_dir}")

        # read science image ids from visit list
        template_ids = []
        # get the values, ignore whitespace
        with open(visit_list_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    template_ids.append(line)

        if not template_ids:
            sys.exit("Visit list is empty.")

        # list for matched FITS paths
        fits_files = []
        # List for image ids
        image_ids = []

        
        for sci_id in template_ids:
            # search for matching file
            matches = sorted(
                warp_dir.glob(f"DIFFEXP-{sci_id}-*-{self.ccd}.fits")
            )
            if len(matches) == 1:
                fits_files.append(matches[0])
                image_ids.append(sci_id)
            elif len(matches) == 0:
                print(f"No DIFFEXP found.")
            else:
                # more than one match?
                print(f"Multiple DIFFEXP matches.")

        if not fits_files:
            sys.exit("No DIFFEXP files found in warp directory.")

        # keep only requested frame count
        fits_files = fits_files[: self.n_frames]
        image_ids = image_ids[: self.n_frames]

        return fits_files, image_ids

    def _load_one(self, fits_file, image_id):
        """Open one FITS file and append data to all lists."""
        with fits.open(fits_file) as hdul:
            # find the first 2D image HDU
            image, img_hdu = self._find_image_hdu(hdul)

            # WCS comes from the image HDU header (HDU 1+)
            img_header = img_hdu.header
            wcs_obj = self._build_wcs(img_header)
            zp = PhotoCalib.get_zeropoint(hdul)
            gain = float(img_header.get("GAIN") or 3.0)

            primary = hdul[0].header
            exptime = float(primary.get("EXPTIME") or
                             img_header.get("EXPTIME") or 0.0)
            if "MJD-OBS" in primary:
                mjd = float(primary["MJD-OBS"])
            elif "MJD-OBS" in img_header:
                mjd = float(img_header["MJD-OBS"])
            else:
                print(f"no MJD-OBS found in {fits_file.name}")
                mjd = 0.0
            
            # compute midpoint time of exposure
            cent_time  = mjd + exptime / (2.0 * 86400.0)
            pscale = self._pixel_scale(wcs_obj)

        # look up the per-frame PSF from dbimages using the science image ID
        # ASK WES
        psf_path = find_psf_file_from_dbimages(image_id, self.ccd)

        # store values
        self.images.append(image)
        self.mjds.append(mjd)
        self.cent_times.append(cent_time)
        self.exptimes.append(exptime)
        self.zeropoints.append(zp)
        self.gains.append(gain)
        self.wcs_list.append(wcs_obj)
        self.pixel_scales.append(pscale)
        self.image_ids.append(image_id)
        self.fits_files.append(fits_file)
        self.psf_files.append(psf_path)

    def _sort_chronological(self):
        """order all lists by ascending cent_time (MJD midpoint)."""
        
        # Create list of indices sorted by exposure midpoint time
        order = sorted(range(len(self.cent_times)), key=lambda k: self.cent_times[k])
        self.images       = [self.images[k]       for k in order]
        self.mjds         = [self.mjds[k]         for k in order]
        self.cent_times   = [self.cent_times[k]   for k in order]
        self.exptimes     = [self.exptimes[k]     for k in order]
        self.zeropoints   = [self.zeropoints[k]   for k in order]
        self.gains        = [self.gains[k]        for k in order]
        self.wcs_list     = [self.wcs_list[k]     for k in order]
        self.pixel_scales = [self.pixel_scales[k] for k in order]
        self.image_ids    = [self.image_ids[k]    for k in order]
        self.fits_files   = [self.fits_files[k]   for k in order]
        self.psf_files    = [self.psf_files[k]    for k in order]

    # dont need self
    @staticmethod
    def _find_image_hdu(hdul):
        """Return data array as float for first 2D image HDU."""
        for hdu in hdul:
            # check if 2d
            if hdu.data is not None and hdu.data.ndim == 2:
                return hdu.data.astype(float), hdu
        sys.exit("No 2D image HDU found in FITS file.")

    @staticmethod
    def _build_wcs(header):
        """
        Build a WCS from a header, RADECSYS handling
        """
        header = header.copy()

        # Astropy warns about RADECSYS; replace with RADESYSa
        if "RADECSYS" in header and "RADESYSa" not in header:
            header["RADESYSa"] = header["RADECSYS"]

        # High order PV terms can break WCS
        bad_pv = [k for k in header.keys()
                  if k.startswith("PV") and "_" in k and
                  int(k.split("_")[1]) >= 5]
        for k in bad_pv:
            del header[k]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wcs_obj = WCS(header, naxis=2)

        return wcs_obj

    @staticmethod
    def _pixel_scale(wcs_obj):
        """Return mean pixel scale in arcsec/pixel."""
        try:
            scales = wcs_obj.proj_plane_pixel_scales()
            # Average the image acis scalles and convert degrees to arcseconds
            return float(np.mean([s.value for s in scales]) * ARCSEC_PER_DEG)
        except Exception:
            return 0.185  # MegaCam pixel scale


###############
# TemplateDataset
# Loads the images that built the subtraction template for one visit/CCD.
####################

class TemplateDataset:
    """
    Reads the *template* visit list

        /arc/projects/classy/visitLists/<visit>/<visit>_template_visit_list.txt
    """

    def __init__(self, visit, ccd):
        self.visit = visit
        self.ccd = ccd

        self.image_ids = []
        self.fits_files = []
        self.psf_files = []
        self.wcs_list = []
        self.cent_times = []
        self.mjds = []
        self.exptimes = []
        self.zeropoints = []
        self.gains = []
        self.pixel_scales = []

    def load(self):
        """Read the template list, match + open each FITS, fill all lists."""
        template_ids = self._read_template_list()

        warp_dir = WARP_ROOT / self.visit / str(self.ccd)
        if not warp_dir.is_dir():
            sys.exit(f"Warp directory not found: {warp_dir}")

        for tmpl_id in template_ids:
            # prefer a matching DIFFEXP in the same warp directory
            matches = sorted(
                warp_dir.glob(f"DIFFEXP-{tmpl_id}-*-{self.ccd}.fits")
            )
            if len(matches) == 0:
                print(f"skipping.")
                continue
            if len(matches) > 1:
                print(f"too many")
            fits_file = matches[0]

            # we need a PSF to render this template's negative well
            psf_path = find_psf_file_from_dbimages(tmpl_id, self.ccd)
            if psf_path is None:
                print(f"No PSF")
                continue

            self._load_one(fits_file, tmpl_id, psf_path)

        if not self.image_ids:
            sys.exit("No loaded.")
        for i, tid in enumerate(self.image_ids):
            print(f"  T[{i}] id={tid}  zp={self.zeropoints[i]:.4f}  "
                  f"exptime={self.exptimes[i]:.1f}s  mjd={self.mjds[i]:.6f}  "
                  f"psf={self.psf_files[i].name}")
        return self

    def _read_template_list(self):
        """Read template image IDs from the template visit list."""
        path = (VISIT_LIST_ROOT / self.visit /
                f"{self.visit}_template_visit_list.txt")
        if not path.exists():
            sys.exit(f"Template visit list not found: {path}")

        ids = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    ids.append(line)

        if not ids:
            sys.exit("Template visit list is empty.")
        return ids

    def _load_one(self, fits_file, image_id, psf_path):
        """Open one template FITS and append its metadata to all lists."""
        # reuse the science loaders' static helpers; we only need metadata,
        # not the template pixels themselves.
        with fits.open(fits_file) as hdul:
            _image, img_hdu = WarpDataset._find_image_hdu(hdul)

            img_header = img_hdu.header
            wcs_obj = WarpDataset._build_wcs(img_header)
            zp = PhotoCalib.get_zeropoint(hdul)
            gain = float(img_header.get("GAIN") or 3.0)

            primary = hdul[0].header
            exptime = float(primary.get("EXPTIME") or
                            img_header.get("EXPTIME") or 0.0)
            if "MJD-OBS" in primary:
                mjd = float(primary["MJD-OBS"])
            elif "MJD-OBS" in img_header:
                mjd = float(img_header["MJD-OBS"])
            else:
                print(f"no MJD-OBS found in {fits_file.name}")
                mjd = 0.0

            cent_time = mjd + exptime / (2.0 * 86400.0)
            pscale = WarpDataset._pixel_scale(wcs_obj)

        self.image_ids.append(image_id)
        self.fits_files.append(fits_file)
        self.psf_files.append(psf_path)
        self.wcs_list.append(wcs_obj)
        self.cent_times.append(cent_time)
        self.mjds.append(mjd)
        self.exptimes.append(exptime)
        self.zeropoints.append(zp)
        self.gains.append(gain)
        self.pixel_scales.append(pscale)


# PhotoCalib
# Handles zeropoint reading and magnitude→count conversion.

class PhotoCalib:
    """
    Magnitude to photometric zeropoint to image counts (ADU)

        zp = -2.5 * log10(calibrationMean) + 31.4 from wikipedia ASK WES
    """
    
    # converting from nJY
    PHOTOCALIB_ZP_OFFSET = 31.4  

    @staticmethod
    def get_zeropoint(hdul):
        """
        Read the zeropoint from a FITS HDU list.
        """
        # primary method: PhotoCalib BinTable
        for hdu in hdul:
            if not hasattr(hdu, "columns"):
                continue
            # make list of all column names and checks if name in it
            col_names = [c.name for c in hdu.columns]
            if "calibrationMean" in col_names:
                cal_mean = float(hdu.data["calibrationMean"][0])
                if cal_mean > 0:
                    zp = -2.5 * np.log10(cal_mean) + PhotoCalib.PHOTOCALIB_ZP_OFFSET
                    return float(zp)

        # simple header keywords
        for hdu in hdul:
            for zpv in ("PHOTZP", "MAGZP", "ZEROPT", "FLXMAG0"):
                val = hdu.header.get(zpv)
                if val:
                    return float(val)

        print("No zeropoint found.")
        return 0.0

    @staticmethod
    def mag_to_counts(mag, zeropoint):
        """
        Convert AB magnitude to expected total count (ADU).
        """
        return 10.0 ** ((zeropoint - mag) / 2.5)


########
# OrbitSampler
# Generates random synthetic TNO orbital parameters.

class OrbitSampler:
    """
    Randomly makes heliocentric distance, eccentricity, and inclination
    for a synthetic KBO.

    alpha_rad = solar elongation angle
    theta_rad = true anomaly. near-opposition default
    """

    # Classical KBO values
    R_MIN = 10.0   # Neptune (AU)
    R_MAX = 20.0   # Kuiper belt edge 
    E_MAX = 0.25
    I_MAX_DEG = 35.0  # degrees

    def __init__(self, r_min=None, r_max=None, e_max=None, i_max_deg=None, rng=None):
        self.r_min = r_min or self.R_MIN
        self.r_max = r_max or self.R_MAX
        self.e_max = e_max or self.E_MAX
        self.i_max_deg = i_max_deg or self.I_MAX_DEG
        self.rng = rng or np.random.default_rng()

    def sample(self):
        """Return one randomly drawn set of orbital parameters as a dict."""
        # randomly generate the calues
        r = self.rng.uniform(self.r_min, self.r_max)
        e = self.rng.uniform(0.0, self.e_max)
        i_deg = self.rng.uniform(0.0, self.i_max_deg)
        i_rad = np.deg2rad(i_deg)

        # Near-opposition approximation Appendix B
        alpha_rad = 0.0
        theta_rad = 0.0

        return {
            "r": r,
            "e": e,
            "i_deg": i_deg,
            "i_rad": i_rad,
            "alpha_rad": alpha_rad,
            "theta_rad": theta_rad,
        }

################
# Converts orbital parameters to sky-plane motion rates.
################

class MotionModel:
    """
    Computes apparent sky-plane motion of a TNO using the Appendix B
    """
    EARTH_ANGULAR_VELOCITY = 148.0   # arcsec/hr

    @staticmethod
    def compute(orbit):
        """
        Returns dict with rate_ra, rate_dec, rate, angle.
        """
        r = orbit["r"]
        i = orbit["i_rad"]
        alpha = orbit["alpha_rad"]
        theta = orbit["theta_rad"]

        # geocentric distance approximation
        delta = r - 1.0
        if delta <= 0.0:
            delta = 0.1   # GET RIF OF DIVISION BY ZERO ERROR

        # Appendix B equations
        raw_x = np.cos(alpha) / delta - np.cos(i) * np.cos(alpha + theta) / (r ** 1.5)
        raw_y = np.sin(i) / (r ** 1.5)

        # convert to arcsec/hr
        rate_ra  =  MotionModel.EARTH_ANGULAR_VELOCITY * raw_x
        rate_dec =  MotionModel.EARTH_ANGULAR_VELOCITY * raw_y

        # combined speed 
        rate  = np.sqrt(rate_ra**2 + rate_dec**2)
        # Position anfle measured from N through E
        angle = np.degrees(np.arctan2(rate_ra, rate_dec)) % 360.0
        
        print(rate)

        return {
            "rate_ra": rate_ra,
            "rate_dec": rate_dec,
            "rate": rate,
            "angle": angle,
        }

##############
# Draw a random apparent magnitude for the synthetic TNO
########

class MagnitudeSampler:
    """
    Draws a random AB magnitude for the fake TNO.
    """

    # Should be 27 but for testing is very bright
    MAG_MIN = 22.0
    MAG_MAX = 24.0

    def __init__(self, mag_min=None, mag_max=None, rng=None):
        self.mag_min = mag_min or self.MAG_MIN
        self.mag_max = mag_max or self.MAG_MAX
        self.rng = rng or np.random.default_rng()

    def sample(self):
        # Randomm
        return float(self.rng.uniform(self.mag_min, self.mag_max))

    
    
    
    
    
    
######################
# Chooses a safe random injection position in the first frame.
##############

class PositionSampler:
    """
    Randomly draws a cutout centre (away from all image edges in every frame)
    and a random object position anywhere inside the frame-0
    cutout. The cutout centre is fixed at frame 0's pixel grid.
    """

    # try multiple times before give up
    MAX_TRIES = 50
    # keep the frame-0 object at least this fraction of half_size inside the cutout edge
    EDGE_PAD_FRAC = 0.1

    def __init__(self, dataset, half_size, rng=None):
        self.dataset = dataset
        self.half_size = half_size
        self.rng = rng or np.random.default_rng()

    # choose one safe random injection position
    def sample(self):
        """
        Returns (ra_ref, dec_ref, x_cut, y_cut) or raises error.

        (x_cut, y_cut) is the cutout centre in frame 0 (edge-safe in all frames).
        (ra_ref, dec_ref) is the object reference, randomly offset from the
        cutout centre so it can sit anywhere inside the frame-0 cutout.
        """
        # get first image, get wcs solution, get h and w in pixels
        img0 = self.dataset.images[0]
        wcs0 = self.dataset.wcs_list[0]
        h, w = img0.shape
        margin = self.half_size
        # how far the object can sit from the cutout centre and stay inside it
        max_off = self.half_size - int(round(self.half_size * self.EDGE_PAD_FRAC))

        for _ in range(self.MAX_TRIES):
            # random cutout centre in frame 0, full cutout away from edges
            cx = self.rng.uniform(margin, w - margin)
            cy = self.rng.uniform(margin, h - margin)

            # the cutout window must be edge-safe in every frame
            ra_c_arr, dec_c_arr = wcs0.all_pix2world([cx], [cy], 0)
            if not self._safe_in_all_frames(float(ra_c_arr[0]), float(dec_c_arr[0])):
                continue

            # random object position anywhere inside the frame-0 cutout
            ox = self.rng.uniform(-max_off, max_off)
            oy = self.rng.uniform(-max_off, max_off)
            x_obj = cx + ox
            y_obj = cy + oy

            ra_arr, dec_arr = wcs0.all_pix2world([x_obj], [y_obj], 0)
            ra  = float(ra_arr[0])
            dec = float(dec_arr[0])

            return ra, dec, float(cx), float(cy)

        raise RuntimeError(
            f"Could not find a safe position ")

    def _safe_in_all_frames(self, ra, dec, margin=None):
        """True if  maps to a valid, edge-safe pixel in every frame."""
        m = margin or self.half_size
        # Loops through each WCS obejct and its matching mage
        for wcs_i, img in zip(self.dataset.wcs_list, self.dataset.images):
            h, w = img.shape
            px_arr, py_arr = wcs_i.all_world2pix([ra], [dec], 0)
            px = float(px_arr[0])
            py = float(py_arr[0])
            if not (m <= px < w - m and m <= py < h - m):
                return False
        return True


#############
# PSFInjector
# Loads a TRIPPy PSF and injects a trailed fake source into images.
################

class PSFInjector:
    """
    Injects trippy trailed fake sources into images.
    """

    # build the trailed line PSF once per frame, reuse it for every well
    def build_line_psf(self, psf_file, rate, motion_rate_ra, motion_rate_dec,
                       exptime, pixel_scale):
        psf_file = Path(psf_file)
        if not psf_file.exists():
            sys.exit(
                f"PSF file not found,")

        # restore PSF fresh each call
        mpsf = trippy_psf.modelPSF(restore=str(psf_file))
        r2d = 180.0 / np.pi
        # Convert ra/dec motion to pixel angle
        trippy_angle = np.arctan2(-motion_rate_dec, motion_rate_ra) * r2d + 180.0

        if trippy_angle>90:
            trippy_angle -= 180
        if trippy_angle<-90:
            trippy_angle+=180.

        mpsf.line(
            rate,                   # total rate in arcsec/hr
            trippy_angle,           # TRIPPy pixel-space angle
            exptime / 3600.0,       # exposure duration in hours
            pixScale=pixel_scale,
            useLookupTable=True,
        )
        return mpsf
    # this is the make_psf_stamp_or_image you asked for. amplitude can be < 0.
    def make_psf_image(self, image_shape, x, y, counts, mpsf, gain,
                       add_poisson_noise):
        A, B = image_shape

        # plant at unit amplitude on a blank canvas
        p_im = mpsf.plant(
            np.array([x]),
            np.array([y]),
            np.array([1.0]),
            np.zeros(image_shape, dtype=float),
            useLinePSF=True,
            returnModel=True,
            gain=1.0,
            addNoise=False,
            verbose=False,
        )
        norm_flux = np.sum(p_im)
        if norm_flux <= 0:
            print("PSF plant returned zeroflux")
            return np.zeros(image_shape, dtype=float), 0.0

        # counts signed: amp inherits the sign for negative wells
        amp = counts / norm_flux
        p_im = p_im * amp
        if add_poisson_noise:
            p_im += np.random.randn(A, B) * np.sqrt(np.abs(p_im) / float(gain))

        return p_im, float(np.sum(p_im))

    def inject(self, image, x, y, rate, motion_rate_ra, motion_rate_dec,
               mag, exptime, zeropoint, gain, pixel_scale, psf_file,
               debug_fake_only=False,
               negative_wells=None,
               add_noise_to_negative_wells=False):
        """
        Build the full fake DIFFEXP contribution for one science frame:
        a positive source from mag/zeropoint using THIS frame's PSF, PLUS the
        negative template wells in `negative_wells`. Each well carries its own
        template PSF file / exptime / pixel scale, so the negative residual is
        rendered with the template image's PSF, not the science frame's.

        Each well is a dict:
            {"x", "y", "counts", "psf_file", "exptime", "pixel_scale", "image_id"}
        """
        # real image copy, or zeroed for debug mode
        if debug_fake_only:
            background = np.zeros_like(image, dtype=float)
        else:
            background = image.copy().astype(float)

        # positive source uses the SCIENCE frame PSF
        sci_mpsf = self.build_line_psf(
            psf_file, rate, motion_rate_ra, motion_rate_dec, exptime, pixel_scale
        )

        # positive source: counts from the magnitude/zeropoint as before
        pos_counts = PhotoCalib.mag_to_counts(mag, zeropoint)
        p_im, pos_flux = self.make_psf_image(
            image.shape, x, y, pos_counts, sci_mpsf, gain,
            add_poisson_noise=True,
        )
        new_image = background + p_im

        # negative wells: each template's contribution, with its OWN PSF.
        if negative_wells:
            for well in negative_wells:
                well_mpsf = self.build_line_psf(
                    well["psf_file"], rate, motion_rate_ra, motion_rate_dec,
                    well["exptime"], well["pixel_scale"],
                )
                w_im, _w_flux = self.make_psf_image(
                    image.shape, well["x"], well["y"], well["counts"],
                    well_mpsf, gain,
                    add_poisson_noise=add_noise_to_negative_wells,
                )
                new_image = new_image + w_im

        return new_image, float(pos_flux)
    
    
    
    
    
    
    
###############
# TensorBuilder
# Extracts cutouts, builds the tensor, saves .npz.
#############

class TensorBuilder:
    """
    Extracts fixed-size cutouts centred on the reference position in
    frame 0 and stacks them into a tensor.

    Cutout centre is always the reference projected into frame 0.
    The object drifts across the cutout in later frames.
    """

    def __init__(self, half_size):
        self.half_size = half_size

    def build(self, injected_images, x_ref, y_ref):
        cutouts = []
        for img in injected_images:
            c = self._cutout(img, x_ref, y_ref)
            if c is None:
                return None
            cutouts.append(c)
        return np.stack(cutouts, axis=0)

    def _cutout(self, image, x, y):
        h, w = image.shape
        cx = int(round(x))
        cy = int(round(y))
        x0, x1 = cx - self.half_size, cx + self.half_size
        y0, y1 = cy - self.half_size, cy + self.half_size
        # check if cutoutisnde inside image boundaries
        if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
            return None
        cutout = image[y0:y1, x0:x1].copy()
        return cutout

    def save(self, out_path, tensor, metadata):
        """Save tensor + metadata dict to a compressed .npz file."""
        np.savez_compressed(
            out_path,
            tensor=tensor,
            **{k: np.asarray(v) for k, v in metadata.items()},
        )

############
# PLOT
##################
class DiagnosticPlotter:
    """Saves a plot for tensor."""

    def __init__(self, half_size):
        self.half_size = half_size

    def save(self, out_path, tensor, positions, x_ref, y_ref,
             image_ids, cent_times, title_info, negative_positions=None):
        
        n = tensor.shape[0]
        hs = self.half_size
        vmin = np.nanpercentile(tensor, 1)
        vmax = np.nanpercentile(tensor, 99)
        t0 = cent_times[0]

        fig, axes = plt.subplots(n, 1, figsize=(4, 4 * n))
        if n == 1:
            axes = [axes]

        for i, ax in enumerate(axes):
            ax.imshow(tensor[i], origin="lower", cmap="gray", vmin=vmin, vmax=vmax)

            # purple cross = fixed reference centre
            ax.plot(hs, hs, "+", color="purple", markersize=8)

            # cyan dot = expected injection position in this frame
            x_inj, y_inj = positions[i]
            x_rel = hs + (x_inj - x_ref)
            y_rel = hs + (y_inj - y_ref)
            ax.plot(x_rel, y_rel, ".", alpha= 0.1, color="cyan", markersize=8)

            dt_hr = (cent_times[i] - t0) * HOURS_PER_DAY
            ax.set_title(
                f"frame {i}  id={image_ids[i]}  dt={dt_hr:.3f} hr"
                f"  inj=({x_inj:.1f},{y_inj:.1f})",
                fontsize=7, loc="left",
            )
            ax.axis("off")

        suptitle = (
            f"mode=fake | CCD={title_info.get('ccd')} | mag={title_info.get('mag'):.2f}\n"
            f"r={title_info.get('r'):.2f} AU  i={title_info.get('i_deg'):.1f} deg  "
            f"rate={title_info.get('rate'):.4f} arcsec/hr  angle={title_info.get('angle'):.1f} deg\n"
        )
        fig.suptitle(suptitle, fontsize=8, y=0.999)
        plt.tight_layout(rect=[0, 0, 1, 1])
        plt.savefig(out_path, dpi=120)
        plt.close(fig)


##################      
# DEBUGGGGG
################


class PlantListHelper:
    """
    Optional helper for comparing WCS-derived positions against plantList
    x/y coordinates, or for avoiding regions near existing implants.

    Not called by SyntheticTNOPipeline in normal operation.
    """

    PLANT_COLS = [
        "index", "ra", "dec", "x", "y",
        "rate", "angle", "rate_ra", "rate_dec",
        "mag", "psf_amp", "g_i",
    ]

    @classmethod
    def load(cls, plant_file):
        """Load a plantList file and return a DataFrame."""
        import pandas as pd
        df = pd.read_csv(
            plant_file,
            sep=r"\s+",
            comment="#",
            names=cls.PLANT_COLS,
            dtype=float,
        )
        df["index"] = df["index"].astype(int)
        return df

    @classmethod
    def implanted_positions(cls, plant_file):
        """Return list of (ra, dec) for all objects in a plantList."""
        df = cls.load(plant_file)
        return list(zip(df["ra"].values, df["dec"].values))


    
    
    
    
    
    
    
############################################
# SyntheticTNOPipeline
# Orchestrates everything.
################################

class SyntheticTNOPipeline:
    """
    Main pipeline.

    1. Load the science warp dataset once.
    2. Load the template dataset once (the real subtraction-template images).
    3. For each sample:
       a. Sample orbit (OrbitSampler)
       b. Compute motion (MotionModel)
       c. Sample magnitude (MagnitudeSampler)
       d. Sample safe injection position (PositionSampler)
       e. Inject fake TNO into each science frame (PSFInjector), with negative
          wells from the template images
       f. Build tensor (TensorBuilder)
       g. Save .npz (TensorBuilder.save)
       h. Save diagnostic plot (DiagnosticPlotter)
    4. Print a summary.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.out_dir = Path(cfg["out_dir"])
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.rng = np.random.default_rng()

        # build sub-components
        self.dataset = WarpDataset(
            visit = cfg["visit"],
            ccd = cfg["ccd"],
            n_frames = cfg["n_frames"],
            sort_by_time = cfg["sort_by_time"],
        )

        # the real subtraction-template images (separate visit list)
        self.template_dataset = TemplateDataset(
            visit = cfg["visit"],
            ccd = cfg["ccd"],
        )

        # mAKE PSF shaped 
        self.psf_injector = PSFInjector()
        # Object that randomly samples fake 
        self.orbit_sampler = OrbitSampler(rng=self.rng)
        # Object for random magnitude
        self.mag_sampler = MagnitudeSampler(rng=self.rng)
        # BUild tensor and plot
        self.tensor_builder = TensorBuilder(half_size=cfg["half_size"])
        self.plotter = DiagnosticPlotter(half_size=cfg["half_size"])

    def run(self):
        self.dataset.load()
        self.template_dataset.load()

        # creates position sampler
        self.pos_sampler = PositionSampler(
            dataset = self.dataset,
            half_size = self.cfg["half_size"],
            rng = self.rng,
        )
        
        # how many fake samples, and intializes fails and yays
        n_samples = self.cfg["num_samples"]
        n_success = 0
        n_fail = 0

        for sample_idx in range(n_samples):

            try:
                result = self._generate_one(sample_idx)
            except RuntimeError as e:
                print(f"{sample_idx}: FAILED — {e}")
                n_fail += 1
                continue

            if result is None:
                n_fail += 1
                continue

            n_success += 1


        print(f"\n Generated {n_success} samples, {n_fail} failures.")
        print(f"Output: {self.out_dir}")

    ##################
    # Generate one sample

    def _generate_one(self, sample_idx):
        """
        Generate one synthetic TNO tensor.
        """
        
        # fake orbital paramters
        orbit = self.orbit_sampler.sample()
        # paramters to apparent sky motion
        motion  = MotionModel.compute(orbit)
        # random magnitude
        mag = self.mag_sampler.sample()

        # pick a safe injection position in frame 0
        ra_ref, dec_ref, x_ref, y_ref = self.pos_sampler.sample()

        dataset = self.dataset
        tmpl = self.template_dataset
        n_tmpl = len(tmpl.image_ids)

        # inject into every frame
        injected_images = []
        injection_positions = []
        injected_fluxes = []
        negative_positions = []   # per science-frame list of (x,y) well positions
        
        
        # midpoint to be used for first exposire
        cent_time0 = dataset.cent_times[0]

        for i in range(len(dataset.images)):
            # Compute time difference from frame 0
            dt_hours = (dataset.cent_times[i] - cent_time0) * HOURS_PER_DAY
            pscale = dataset.pixel_scales[i]
            wcs_i = dataset.wcs_list[i]

            # expected pixel displacement from motion
            dx_pix =  motion["rate_ra"]  * dt_hours / pscale
            dy_pix = -motion["rate_dec"] * dt_hours / pscale   

            # reference position in this frame via WCS
            # do the dx dy thing here 
            x_ref_i_arr, y_ref_i_arr = wcs_i.all_world2pix([ra_ref], [dec_ref], 0)
            x_ref_i = float(x_ref_i_arr[0])
            y_ref_i = float(y_ref_i_arr[0])

            x_inj = x_ref_i + dx_pix
            y_inj = y_ref_i + dy_pix

            frame_psf = dataset.psf_files[i]
            if frame_psf is None:
                sys.exit(
                    f"No PSF found for image_id")

            negative_wells = []
            for j in range(n_tmpl):
                dt_j = (tmpl.cent_times[j] - cent_time0) * HOURS_PER_DAY
                xj, yj = project_track_position(
                    wcs_i, pscale, ra_ref, dec_ref, motion, dt_j
                )
                A_j = PhotoCalib.mag_to_counts(mag, tmpl.zeropoints[j])
                negative_wells.append({
                    "x":           xj,
                    "y":           yj,
                    "counts":      -A_j / n_tmpl,
                    "psf_file":    tmpl.psf_files[j],
                    "exptime":     tmpl.exptimes[j],
                    "pixel_scale": tmpl.pixel_scales[j],
                    "image_id":    tmpl.image_ids[j],
                })

            # In ject the fake object into the current image frame
            new_img, inj_flux = self.psf_injector.inject(
                image = dataset.images[i],
                x = x_inj,
                y = y_inj,
                rate = motion["rate"],
                motion_rate_ra = motion["rate_ra"],
                motion_rate_dec = motion["rate_dec"],
                mag = mag,
                exptime = dataset.exptimes[i],
                zeropoint = dataset.zeropoints[i],
                gain = dataset.gains[i],
                pixel_scale = pscale,
                psf_file = frame_psf,
                debug_fake_only = self.cfg["debug_fake_only"],
                negative_wells = negative_wells,
                add_noise_to_negative_wells =
                    self.cfg.get("add_noise_to_negative_wells", False),
            )

            
            # Store the image with the fake object and measurements
            injected_images.append(new_img)
            injection_positions.append((x_inj, y_inj))
            negative_positions.append([(w["x"], w["y"]) for w in negative_wells])
            print("frame", i)
            for w in negative_wells:
                print(w["x"], w["y"], w["counts"], "tmpl", w["image_id"])
            injected_fluxes.append(inj_flux)

        # build tensor using the frame 0 reference pixel as the cutout centre
        tensor = self.tensor_builder.build(injected_images, x_ref, y_ref)

        if tensor is None:
            print(f"Cutout out of bounds.")
            return None

        template_psf_files = [str(p) for p in tmpl.psf_files]

        #  metadata dictionary
        metadata = {
            # orbital parameters
            "r": orbit["r"],
            "e": orbit["e"],
            "i_deg": orbit["i_deg"],
            "alpha_rad": orbit["alpha_rad"],
            "theta_rad": orbit["theta_rad"],
            # motion
            "rate_ra":motion["rate_ra"],
            "rate_dec": motion["rate_dec"],
            "rate": motion["rate"],
            "angle": motion["angle"],
            # photometry
            "mag": mag,
            # injection geometry
            "ra_ref":ra_ref,
            "dec_ref":dec_ref,
            "x_ref": x_ref,
            "y_ref": y_ref,
            "positions": np.array(injection_positions),
            # negative wells (n_science_frames, n_template, 2)
            "negative_positions": np.array(negative_positions),
            "n_template": n_tmpl,
            # timing
            "mjds": np.array(dataset.mjds),
            "cent_times": np.array(dataset.cent_times),
            "exptimes": np.array(dataset.exptimes),
            "zeropoints": np.array(dataset.zeropoints),
            # dataset info
            "visit": self.cfg["visit"],
            "ccd": self.cfg["ccd"],
            "image_ids": np.array(dataset.image_ids),
            # template info
            "template_image_ids": np.array(tmpl.image_ids),
            "template_zeropoints": np.array(tmpl.zeropoints),
            "template_exptimes": np.array(tmpl.exptimes),
            "template_cent_times": np.array(tmpl.cent_times),
            "template_mjds": np.array(tmpl.mjds),
            "template_psf_files": np.array(template_psf_files),
            "sample_idx": sample_idx,
            # debug
            "debug_fake_only": int(self.cfg["debug_fake_only"]),
            "injected_fluxes": np.array(injected_fluxes),
        }

        # save .npz
        run_label = (
            f"{self.cfg['visit']}_CCD{self.cfg['ccd']}_"
            f"S{sample_idx:06d}"
        )
        out_npz = self.out_dir / f"tensor_{run_label}.npz"
        self.tensor_builder.save(out_npz, tensor, metadata)

        # optional plot
        if self.cfg["save_plots"]:
            out_png = self.out_dir / f"tensor_{run_label}.png"
            title_info = {
                "ccd": self.cfg["ccd"],
                "mag": mag,
                "r": orbit["r"],
                "i_deg": orbit["i_deg"],
                "rate":motion["rate"],
                "angle": motion["angle"],
            }
            self.plotter.save(
                out_png, tensor,
                injection_positions,
                x_ref, y_ref,
                dataset.image_ids,
                dataset.cent_times,
                title_info,
                negative_positions,

            )

        return {
            "mag": mag,
            "r": orbit["r"],
            "rate": motion["rate"],
            "mean_injected_flux": float(np.mean(injected_fluxes)),
        }


    
#################    
 #################
    
#MAIN FUNCTION#
    
#################
#################

def main():
    cfg = parse_args()

    print("=" * 60)
    print("Synthetic TNO Dataset Generator")
    print("=" * 60)
    print(f"  Visit:       {cfg['visit']}")
    print(f"  CCD:         {cfg['ccd']}")
    print(f"  PSF source:  dbimages (auto-lookup per frame)")
    print(f"  Templates:   {cfg['visit']}_template_visit_list.txt")
    print(f"  Num samples: {cfg['num_samples']}")
    print(f"  Frames:      {cfg['n_frames']}")
    print(f"  Half-size:   {cfg['half_size']} px  "
          f"(cutout = {2*cfg['half_size']}x{2*cfg['half_size']} px)")
    print(f"  Output dir:  {cfg['out_dir']}")
    print(f"  Sort by MJD: {cfg['sort_by_time']}")
    print(f"  Debug (zero bg): {cfg['debug_fake_only']}")
    print(f"  Save plots:  {cfg['save_plots']}")
    print("=" * 60)

    pipeline = SyntheticTNOPipeline(cfg)
    pipeline.run()


if __name__ == "__main__":
    main()
