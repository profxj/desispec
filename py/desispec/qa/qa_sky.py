"""
desispec.qa.qs_sky
============

Utility functions to do simple QA 
"""

import numpy as np
from desispec.resolution import Resolution
from desispec.log import get_logger
from desispec import util
from desispec.io import read_fiberflat

from desispec.io import frame as desi_io_frame
from desispec.io import fibermap as desi_io_fmap
from desispec.io import read_sky
from desispec import sky as dspec_sky

from desispec.fiberflat import apply_fiberflat

import scipy,scipy.sparse
import sys, os
import pdb

def tst_meansky_fibers(simspec_fil, frame_root, fflat_root, path=None):
    '''Examines mean sky in SKY fibers
    '''
    # imports
    import specter.throughput as spec_thru
    import specter.psf.spotgrid as SpotGridPSF
    sys.path.append(os.path.abspath("/Users/xavier/DESI/desisim_v0.4.1/desisim/"))
    import interpolation as desi_interp
    import io as desisim_io
    #

    from astropy.io import fits
    from xastropy.xutils import xdebug as xdb

    # Truth from simspec
    simsp_hdu = fits.open(simspec_fil)
    hdu_names = [hdu.name for hdu in simsp_hdu]

    # DESI extraction
    # Load
    if path is None:
        path=''
    fiber_fil='fibermap'+frame_root[frame_root.find('-'):]
    fiber_map = desi_io_fmap.read_fibermap(path+fiber_fil)

    # Loop on Camera
    #for camera in ['R','Z']:
    for camera in ['B','R','Z']:
        # TRUTH FIRST
        # Grab header
        idx = hdu_names.index('SKYPHOT_'+camera)
        hdr = simsp_hdu[idx].header
        # Generate wavelength array
        wave = hdr['CRVAL1'] + (hdr['CDELT1'])*np.arange(hdr['NAXIS1'])
        # Grab data
        sky_truth = simsp_hdu[idx].data

        # Get model
        frame_fil = 'frame-{:s}'.format(camera.lower())+frame_root
        frame = desi_io_frame.read_frame(path+frame_fil)

        # Flat field
        fflat_fil = 'fiberflat-{:s}'.format(camera.lower())+fflat_root

        # read fiberflat
        fiberflat = read_fiberflat(path+fflat_fil)

        # apply fiberflat to sky fibers
        apply_fiberflat(frame, fiberflat)

        # Generate sky
        #skymodel, skyflux, skyvar = dspec_sky.compute_sky(obs_frame,fiber_map)

        # Read sky
        sky_fil = 'sky-{:s}'.format(camera.lower())+frame_root
        skymodel=read_sky(path+sky_fil)
        # subtract sky
        dspec_sky.subtract_sky(frame, skymodel)

        # Isolate SKY fibers
        skyfibers = np.where(fibermap["OBJTYPE"]=="SKY")[0]
        skyfibers = skyfibers[skyfibers<500]

        xdb.set_trace()
        # Generate QA
        dspec_sky.make_model_qa(skymodel.wave, skyflux, skyvar, 
            wave, sky_truth, outfil='QA_sky_mean_'+camera+'.pdf')
        # 
        #xdb.set_trace()

def tst_deconvolve_mean_sky(simspec_fil, frame_root, fiber_fil, path=None):
    '''Compares deconvolved sky against Truth
    -- A bit goofy.  
    DEPRECATED
    '''
    # imports
    import specter.throughput as spec_thru
    import specter.psf.spotgrid as SpotGridPSF
    sys.path.append(os.path.abspath("/Users/xavier/DESI/desisim_v0.4.1/desisim/"))
    import interpolation as desi_interp
    import io as desisim_io
    #

    from astropy.io import fits
    from xastropy.xutils import xdebug as xdb

    # Truth from simspec
    simsp_hdu = fits.open(simspec_fil)
    hdu_names = [hdu.name for hdu in simsp_hdu]

    # DESI extraction
    # Load
    if path is None:
        path=''
    fiber_map = desi_io_fmap.read_fibermap(path+fiber_fil)

    # Loop on Camera
    #for camera in ['R','Z']:
    for camera in ['B','R','Z']:
        # TRUTH FIRST
        # Grab header
        idx = hdu_names.index('SKYPHOT_'+camera)
        hdr = simsp_hdu[idx].header
        # Generate wavelength array
        wave = hdr['CRVAL1'] + (hdr['CDELT1'])*np.arange(hdr['NAXIS1'])
        # Grab data
        sky_truth = simsp_hdu[idx].data

        # Get model
        frame_fil = 'frame-{:s}'.format(camera.lower())+frame_root
        obs_frame = desi_io_frame.read_frame(path+frame_fil)
        skymodel, skyflux, skyvar = dspec_sky.compute_sky(obs_frame,fiber_map)

        # Generate QA
        dspec_sky.make_model_qa(skymodel.wave, skyflux, skyvar, 
            wave, sky_truth, outfil='QA_sky_mean_'+camera+'.pdf')
        # 
        #xdb.set_trace()

# ##################################################
# ##################################################
# ##################################################
# Command line execution for testing
# ##################################################
if __name__ == '__main__':

    flg_tst = 0 
    #flg_tst += 2**0  # Deconvolved mean [DEPRECATED]
    flg_tst += 2**1  # Sky fiber mean 

    if (flg_tst % 2**1) >= 2**0:
        path = '/Users/xavier/DESI/TST/20150211/' 
        tst_deconvolve_mean_sky('/Users/xavier/DESI/TST/20150211/simspec-00000002.fits',
            '0-00000002.fits', 'fibermap-00000002.fits',
            path=path)

    if (flg_tst % 2**2) >= 2**1:
        path = '/Users/xavier/DESI/TST/20150211/' 
        tst_meansky_fibers('/Users/xavier/DESI/TST/20150211/simspec-00000002.fits',
            '0-00000002.fits', '0-00000001.fits', path=path)
