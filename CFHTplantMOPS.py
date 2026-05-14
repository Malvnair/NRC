from astropy.io import fits
from astropy import wcs as WCS, time as TIME
from astropy.modeling import functional_models as apmodels
import glob,pickle,os,sys
import time
import numpy as np
import fakeKBOs
from trippy import psf
import multiprocessing as multip
from fluxMagFix import getFluxMag0_ZPT
import argparse

"""
Note about converting from r to gri:

From Ofek 2012, the locus of TNO colours has

(r-i) = 0.8*(g-r) - 0.25

(g-r) + (r-i) = (g-r) + 0.8*(g-r) - 0.25
(g-i) = 1.8*(g-r) - 0.25
(g-r) = 0.555*(g-i) + 0.137

From Megapipe we get a conversion of r_sdss to r_MP of:

(r_MP - r_sdss) = 0.087*(g-r)_sdss = 0.048*(g-i)_sdss + 0.012


Also from Megapipe:

(gri_MP - r_sdss) = -0.0068 + 0.2240*(g-i)_sdss - 0.0563*(g-i)_sdss**2




"""

d2r = np.pi/180.0
r2d = 180.0/np.pi

def planter(i, image_fn, plant_fn, plantListFN, datas, headers, mjd, zeropoints, EXPTIME, fKBOs):
    ccd = f'ccd{str(i).zfill(2)}'
    ccd_str = str(i).zfill(2)



    listHan = open(plantListFN, 'w+')
    print('#index ra dec x y rate ("/hr) angle (deg) rate_ra rate_dec mag psf_amp (g-i)', file=listHan)

    
    wcs = WCS.WCS(headers[i])


    (cent_ra, cent_dec) = wcs.all_pix2world(datas[i].shape[1]/2., datas[i].shape[0]/2., 0)
    plant_im = datas[i]*0.0

    
    #load the trippy psf
    s = image_fn.split('/')
    s.insert(-1, ccd)
    s[-1] = s[-1].replace('p.fits',f'p{ccd_str}.psf.fits')
    psf_filename = "/".join(s)

    mpsf = psf.modelPSF(restore=psf_filename)

    #plant each source
    for j in range(len(fKBOs)):
        
        (index, ra, dec, r_sdss, rr, rd, gmi) = fKBOs[j]

        # convert from r_sdss to megacam r or gri. See note at top of program.
        if headers[i]['FILTER'] == 'gri.MP9605':
            mag = -0.0068 + 0.2240*gmi - 0.0563*gmi*gmi + r_sdss
        elif headers[i]['FILTER'] == 'r.MP9602':
            mag = 0.048*gmi + 0.012 + r_sdss

        if abs(ra-cent_ra)>0.3 or abs(dec-cent_dec)>0.3:
            continue
        try:
            (x, y) = wcs.all_world2pix(ra,dec,0)
            (x1, y1) = wcs.all_world2pix(ra+rr/(3600.0*np.cos(dec*d2r)), dec+rd/3600.0, 0) 
        except:
            (x, y) = -1, -1
            (x1, y1) = -1, -1

        if (x>0 and y>0 and x<B and y<A) and (x1>0 and y1>0 and x1<B and y1<A):
            dx = np.cos(dec*d2r)*rr
            rate = (dx**2 + rd**2)**0.5
            angle = np.arctan2(y1-y, x1-x)*r2d + 180.0
            mpsf.line(rate, angle, EXPTIME/3600., pixScale=0.185, useLookupTable=True)
            p_im = mpsf.plant(np.array([x]),np.array([y]),np.array([1.0]),datas[i]*0.0,useLinePSF=True,returnModel=True,gain=1.0,addNoise = False,verbose=False)
            
            #The below three lines should replace the function above to implant gaussian psfs instead of trippy psfs. The only thing that is needed before using the gaussian psf is that the FWHM needs to be defined.
            #yin, xin = np.indices(datas[i].shape)
            #gauss = apmodels.Gaussian2D(amplitude=1,x_mean=x,y_mean=y,x_stddev=FWHM/2.355,y_stddev=FWHM/2.355)
            #p_im = gauss(xin,yin)

            norm_flux = np.sum(p_im)
            
            amp = (10.0**((zeropoints[i]-mag)/2.5))/(norm_flux)

            gain = headers[i]['GAIN']
            p_im *= amp
            p_im+=np.random.randn(A,B)*np.sqrt(np.abs(p_im)/float(gain) )
            
            print("{}  {:11.6f} {:11.6f} {:7.2f} {:7.2f} {:4.2f} {:6.2f} {:4.2f} {:4.2f} {:5.2f} {:.5f} {:5.2f}".format(index, ra, dec, x, y, rate, angle, rr, rd, mag, amp, gmi))
            print("{}  {:11.6f} {:11.6f} {:7.2f} {:7.2f} {:4.2f} {:6.2f} {:4.2f} {:4.2f} {:5.2f} {:.5f} {:5.2f}".format(index, ra, dec, x, y, rate, angle, rr, rd, mag, amp, gmi),file=listHan)
            
            plant_im += p_im
            #print(np.sum(plant_im))

    listHan.close()
    return np.copy(plant_im)



parser = argparse.ArgumentParser()
parser.add_argument('image_n', help="db image number.", default='2773082')
parser.add_argument('--newvisit', help = 'Specify the plant file to open. This should correspond to the mjd specified by mjd.', default=None)
parser.add_argument('--mjd', help = 'Specify a different MJD to plant with.', default=None)
parser.add_argument('--dbimages-dir', default = '/arc/projects/classy/dbimages', help = 'The dimages directory to read images and save planted images to. DEFAULT=%(default)s')
parser.add_argument('--dbimages-saves-dir', default = '/arc/projects/classy/dbimages', help = 'The db images location to save the planted file to. Can alter to place in scratch. DEFAULT=%(default)s')
args = parser.parse_args()

    
if args.mjd is not None:
    print(f'Altering the output header to use user provided mjd: {args.mjd}')


image_n = args.image_n
dbimages_dir = args.dbimages_dir+'/'
dbimages_saves_dir = args.dbimages_saves_dir

image_fn = f'{dbimages_dir}/{image_n}/{image_n}p.fits'
plant_fn = f'{dbimages_saves_dir}/{image_n}/fk{image_n}p.fits' if args.mjd is None else f'{dbimages_saves_dir}/{image_n}/fk{image_n}s.fits'


print('Loading headers...')
headers = []
datas = []
with fits.open(image_fn) as han:
    EXPTIME = han[0].header['EXPTIME']
    mjd = han[0].header['MJD-OBS'] if args.mjd is None else float(args.mjd)
    h0 = han[0].header
    if args.mjd is not None: # to allow time randomization
        h0['MJD-OBS']=mjd
    for i in range(1,41):
        headers.append(han[i].header)
        headers[i-1]['CTYPE1']='RA---TPV'
        headers[i-1]['CTYPE2']='DEC--TPV'
        if args.mjd is not None: # to allow time randomization
            headers[i-1]['MJD-OBS'] = mjd
        datas.append(han[i].data)
    A,B = han[1].data.shape



jd = mjd + 2400000.5

zeropoints = []
for i,h in enumerate(headers):
    zeropoints.append(h['PHOTZP'])

if args.newvisit is None:
    makefake_fn = f'{image_n}/{image_n}p.makefake'
else:
    makefake_fn = f'{args.newvisit}/{args.newvisit}p.makefake'
print(f'Opening fake KBO positions file {makefake_fn}.')
with open(f'{dbimages_dir}/{makefake_fn}') as han:
    plant_data = han.readlines()

fKBOs = []
for i in range(1, len(plant_data)):
    s = plant_data[i].split()
    ind = int(float(s[0]))
    ra = float(s[1])
    dec = float(s[2])
    mag = float(s[4])
    rate_ra = float(s[12])
    rate_dec = float(s[13])
    r_sdss = float(s[15])
    fKBOs.append([ind, ra, dec, mag, rate_ra, rate_dec, r_sdss])
fKBOs = np.array(fKBOs)

PrimaryHDU = fits.PrimaryHDU([], header=h0)
HDUs = [PrimaryHDU]

for i in range(40):
    if args.newvisit is None:
        plantListFN = f'{dbimages_saves_dir}/{image_n}/ccd{str(i).zfill(2)}/{image_n}p{str(i).zfill(2)}.plantList'
    else:
        plantListFN = f'{dbimages_saves_dir}/{image_n}/ccd{str(i).zfill(2)}/{image_n}s{str(i).zfill(2)}.plantList'

    p_data = planter(i, image_fn, plant_fn, plantListFN, datas, headers, mjd, zeropoints, EXPTIME, fKBOs)

    HDUs.append(fits.ImageHDU((p_data+datas[i]).astype('int'), header=headers[i]))
    HDUs[-1].scale(type='int16')#, bscale=headers[i]['BSCALE'], bzero=headers[i]['BZERO'])
List = fits.HDUList(HDUs)
print(f'Writing planted image to {plant_fn}')
List.writeto(plant_fn, overwrite=True)

