from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import healpy as hp

from scipy.special import legendre

from .utils import dl2cl, print_warning, arrdim 


def getcov_est(cls, nside=4, pls=None, lmax=None, nsample=100, isDl=False, rseed=42, pol=False):
    if (lmax==None):
        lmax = nside*3-1

    if (isDl):
        cls = dl2cl(cls)
    else:
        cls = cls.copy()

    if (type(pls)!=type(None)):
        print_warning("\'pls\' option for getcov_est() is a dummy.")

    ell = np.arange(lmax+1)

    if (pol==True):
        cov = getcov_est_pol(cls, nside, pls=None, lmax=lmax, nsample=nsample, isDl=False, rseed=rseed)
        return cov

    maparr = []
    np.random.seed(rseed)
    for i in range(nsample):
        mapT = hp.synfast(cls, nside=nside, new=True, verbose=False)[0]
        maparr.append(mapT)

    maparr = (np.array(maparr)).T
    cov = np.cov(maparr)

    return cov


def getcov_est_pol(cls, nside=4, pls=None, lmax=None, nsample=100, rseed=42, isDl=False):
    if (lmax==None):
        lmax = nside*3-1

    if (isDl):
        cls = dl2cl(cls)
    else:
        cls = cls.copy()

    if (type(pls)!=type(None)):
        print_warning("\'pls\' option for getcov_est_pol() is a dummy.")

    ell = np.arange(lmax+1)

    maparr = []
    np.random.seed(rseed)
    for i in range(nsample):
        maps = hp.synfast(cls, nside=nside, new=True, verbose=False, pol=True)
        mapTQU = maps.reshape(-1) 
        maparr.append(mapTQU)

    maparr = (np.array(maparr)).T
    cov = np.cov(maparr)

    return cov


def getvar_est(cls, nside=4, Nsample=1000, isDl=False):
    lmax = 3*nside-1
    ells = np.arange(lmax+1)

    if (isDl):
        cls_TT = np.zeros(len(cls))
        cls_TT[1:] = cls[1:] / ells[1:] / (ells[1:]+1) * 2 * np.pi
    else:
        cls_TT = cls.copy()

    vararr = []
    for i in xrange(Nsample):
        np.random.seed(i)
        mapT = hp.synfast(cls_TT, nside=nside, verbose=False)
        vararr.append(np.var(mapT))

    return vararr



