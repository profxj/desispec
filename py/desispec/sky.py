"""
desispec.sky
============

Utility functions to compute a sky model and subtract it.
"""


import numpy as np
from desispec.resolution import Resolution
from desispec.linalg import cholesky_solve
from desispec.linalg import cholesky_solve_and_invert
from desispec.linalg import spline_fit
from desispec.log import get_logger
from desispec import util

import scipy,scipy.sparse
import sys

def compute_sky(frame, fibermap, nsig_clipping=4.) :
    """Compute a sky model.

    Input has to correspond to sky fibers only.
    Input flux are expected to be flatfielded!
    We don't check this in this routine.

    args:
        frame : Frame object, which includes attributes
          - wave : 1D wavelength grid in Angstroms
          - flux : 2D flux[nspec, nwave] density
          - ivar : 2D inverse variance of flux
          - mask : 2D inverse mask flux (0=good)
          - resolution_data : 3D[nspec, ndiag, nwave]  (only sky fibers)
        fibermap : numpy table including OBJTYPE to know which fibers are SKY
        nsig_clipping : [optional] sigma clipping value for outlier rejection

    returns SkyModel object with attributes wave, flux, ivar, mask
    """

    log=get_logger()
    log.info("starting")

    skyfibers = np.where(fibermap["OBJTYPE"]=="SKY")[0]
    #skyfibers = skyfibers[skyfibers<500]

    nwave=frame.nwave
    nfibers=len(skyfibers)
    
    current_ivar=frame.ivar[skyfibers].copy()
    flux = frame.flux[skyfibers]
    Rsky = frame.R[skyfibers]

    sqrtw=np.sqrt(current_ivar)
    sqrtwflux=sqrtw*flux

    chi2=np.zeros(flux.shape)

    #debug
    #nfibers=min(nfibers,2)

    nout_tot=0
    for iteration in range(20) :

        A=scipy.sparse.lil_matrix((nwave,nwave)).tocsr()
        B=np.zeros((nwave))
        # diagonal sparse matrix with content = sqrt(ivar)*flat of a given fiber
        SD=scipy.sparse.lil_matrix((nwave,nwave))
        # loop on fiber to handle resolution
        for fiber in range(nfibers) :
            if fiber%10==0 :
                log.info("iter %d fiber %d"%(iteration,fiber))
            R = Rsky[fiber]

            # diagonal sparse matrix with content = sqrt(ivar)
            SD.setdiag(sqrtw[fiber])

            sqrtwR = SD*R # each row r of R is multiplied by sqrtw[r]

            A = A+(sqrtwR.T*sqrtwR).tocsr()
            B += sqrtwR.T*sqrtwflux[fiber]

        log.info("iter %d solving"%iteration)

        skyflux=cholesky_solve(A.todense(),B)

        log.info("iter %d compute chi2"%iteration)

        for fiber in range(nfibers) :

            S = Rsky[fiber].dot(skyflux)
            chi2[fiber]=current_ivar[fiber]*(flux[fiber]-S)**2

        log.info("rejecting")

        nout_iter=0
        if iteration<1 :
            # only remove worst outlier per wave
            # apply rejection iteratively, only one entry per wave among fibers
            # find waves with outlier (fastest way)
            nout_per_wave=np.sum(chi2>nsig_clipping**2,axis=0)
            selection=np.where(nout_per_wave>0)[0]
            for i in selection :
                worst_entry=np.argmax(chi2[:,i])
                current_ivar[worst_entry,i]=0
                sqrtw[worst_entry,i]=0
                sqrtwflux[worst_entry,i]=0
                nout_iter += 1

        else :
            # remove all of them at once
            bad=(chi2>nsig_clipping**2)
            current_ivar *= (bad==0)
            sqrtw *= (bad==0)
            sqrtwflux *= (bad==0)
            nout_iter += np.sum(bad)

        nout_tot += nout_iter

        sum_chi2=float(np.sum(chi2))
        ndf=int(np.sum(chi2>0)-nwave)
        chi2pdf=0.
        if ndf>0 :
            chi2pdf=sum_chi2/ndf
        log.info("iter #%d chi2=%f ndf=%d chi2pdf=%f nout=%d"%(iteration,sum_chi2,ndf,chi2pdf,nout_iter))

        if nout_iter == 0 :
            break

    log.info("nout tot=%d"%nout_tot)


    # solve once again to get deconvolved sky variance
    skyflux,skycovar=cholesky_solve_and_invert(A.todense(),B)

    #- sky inverse variance, but incomplete and not needed anyway
    skyvar=np.diagonal(skycovar)
    # skyivar=(skyvar>0)/(skyvar+(skyvar==0))

    # Use diagonal of skycovar convolved with mean resolution of all fibers
    # first compute average resolution
    mean_res_data=np.mean(frame.resolution_data,axis=0)
    R = Resolution(mean_res_data)
    # compute convolved sky and ivar
    cskycovar=R.dot(skycovar).dot(R.T.todense())
    cskyvar=np.diagonal(cskycovar)
    cskyivar=(cskyvar>0)/(cskyvar+(cskyvar==0))
    
    # convert cskyivar to 2D; today it is the same for all spectra,
    # but that may not be the case in the future
    cskyivar = np.tile(cskyivar, frame.nspec).reshape(frame.nspec, nwave)

    # Convolved sky
    cskyflux = np.zeros(frame.flux.shape)
    for i in range(frame.nspec):
        cskyflux[i] = frame.R[i].dot(skyflux)

    # need to do better here
    mask = (cskyivar==0).astype(np.uint32)

    # ###
    # Check sky flux here
    from scipy.stats import chisqprob

    qa_dict = {}
    qa_dict['SKYSUB'] = {}
    qa_dict['SKYSUB']['PCHI_FIB'] = 0.95

    # Subtract
    res = flux - cskyflux # Residuals
    res_ivar = util.combine_ivar(current_ivar, cskyivar) 

    # Chi^2
    chi2_fiber = np.zeros(nfibers)
    chi2_prob = np.zeros(nfibers)
    for ii in range(nfibers):
        # Stats
        chi2_fiber[ii] = np.sum(res_ivar*res[:,ii]) 
        dof = np.sum(res_ivar > 0.)-1
        chi2_prob[ii] = chisqprob(chi2_fiber[ii], dof)
        import pdb
        pdb.set_trace()

    import pdb
    pdb.set_trace()

    return SkyModel(frame.wave.copy(), cskyflux, cskyivar, mask)#, skyflux, skyvar

class SkyModel(object):
    def __init__(self, wave, flux, ivar, mask, header=None):
        """Create SkyModel object
        
        Args:
            wave  : 1D[nwave] wavelength in Angstroms
            flux  : 2D[nspec, nwave] sky model to subtract
            ivar  : 2D[nspec, nwave] inverse variance of the sky model
            mask  : 2D[nspec, nwave] 0=ok or >0 if problems
            header : (optional) header from FITS file HDU0
            
        All input arguments become attributes
        """
        assert wave.ndim == 1
        assert flux.ndim == 2
        assert ivar.shape == flux.shape
        assert mask.shape == flux.shape
        
        self.nspec, self.nwave = flux.shape
        self.wave = wave
        self.flux = flux
        self.ivar = ivar
        self.mask = mask
        self.header = header


def subtract_sky(frame, skymodel) :
    """Subtract skymodel from frame, altering frame.flux, .ivar, and .mask
    """
    assert frame.nspec == skymodel.nspec
    assert frame.nwave == skymodel.nwave

    log=get_logger()
    log.info("starting")

    # check same wavelength, die if not the case
    if not np.allclose(frame.wave, skymodel.wave):
        message = "frame and sky not on same wavelength grid"
        log.error(message)
        raise ValueError(message)

    frame.flux -= skymodel.flux
    frame.ivar = util.combine_ivar(frame.ivar, skymodel.ivar)
    frame.mask |= skymodel.mask

    log.info("done")

def make_model_qa(wave, sky_model, sky_var, true_wave, true_sky, 
    frac_res=False, outfil=None):
    """
    Generate QA plots and files
    Parameters:
    true_wave, true_sky:  ndarrays
      Photons/s from simspec file
    """
    from astropy.io import fits
    from matplotlib import pyplot as plt
    import matplotlib.gridspec as gridspec
    import sys, os
    #
    sys.path.append(os.path.abspath("/Users/xavier/DESI/desisim_v0.4.1/desisim/"))
    import interpolation as desi_interp


    # Mean spectrum
    if outfil is None:
        outfil = 'tmp_qa_mean_sky.pdf'

    # Resample
    dw = wave[1]-wave[0]
    ww = wave[0] + np.arange(len(wave))*dw
    true_flux = desi_interp.resample_flux(ww, true_wave, true_sky)
    #import pdb
    #pdb.set_trace()

    # Scale
    scl = np.median(sky_model/true_flux)
    print('scale = {:g}'.format(scl))

    # Error
    sky_sig = np.sqrt(sky_var)

    # Plot
    fig = plt.figure(figsize=(8, 5.0))
    gs = gridspec.GridSpec(3,1)

    xmin,xmax = np.min(wave), np.max(wave)
    # Simple spectrum plot
    ax_flux = plt.subplot(gs[0])
    ax_flux.plot(wave, sky_model, label='Model')
    #ax_flux.plot(wave, sky_sig, label='Model Error')
    ax_flux.plot(wave,true_flux*scl, label='Truth')
    ax_flux.get_xaxis().set_ticks([]) # Suppress labeling
    ax_flux.set_ylabel('Counts')
    ax_flux.set_xlim(xmin,xmax)
    ax_flux.text(0.5, 0.85, 'Sky Meanspec',
        transform=ax_flux.transAxes, ha='center')

    # Legend
    legend = ax_flux.legend(loc='upper right', borderpad=0.3,
                        handletextpad=0.3)#, fontsize='small')

    # Residuals
    scatt_sz = 0.5
    ax_res = plt.subplot(gs[1])
    ax_res.get_xaxis().set_ticks([]) # Suppress labeling
    res = (sky_model - (true_flux*scl))/(true_flux*scl)
    rms = np.sqrt(np.sum(res**2)/len(res))
    #ax_res.set_ylim(-3.*rms, 3.*rms)
    ax_res.set_ylim(-2, 2)
    ax_res.set_ylabel('Frac Res')
    # Error
    #ax_res.plot(true_wave, 2.*ms_sig/sky_model, color='red')
    ax_res.scatter(wave,res, marker='o',s=scatt_sz)
    ax_res.plot([xmin,xmax], [0.,0], 'g-')
    ax_res.set_xlim(xmin,xmax)

    # Relative to error
    ax_sig = plt.subplot(gs[2])
    ax_sig.set_xlabel('Wavelength')
    sig_res = (sky_model - (true_flux*scl))/sky_sig
    ax_sig.scatter(wave, sig_res, marker='o',s=scatt_sz)
    ax_sig.set_ylabel(r'Res $\delta/\sigma$')
    ax_sig.set_ylim(-5., 5.)
    ax_sig.plot([xmin,xmax], [0.,0], 'g-')
    ax_sig.set_xlim(xmin,xmax)


    # Finish
    plt.tight_layout(pad=0.1,h_pad=0.0,w_pad=0.0)
    plt.savefig(outfil)
