from __future__ import print_function
import os

import numpy as np
import healpy as hp

from .utils import dl2cl


def gen_ylm(nside=4, use_weights=True):
    npix = 12 * nside * nside
    oop  = float(npix)/(4*np.pi) # 1/(solid angle of a pixel) 
    marr = np.eye(npix) * oop # delta function map array

    ylm = [hp.map2alm(m, use_weights=use_weights, iter=100) for m in marr]
    return np.array(ylm)


def gen_wxlm(nside=4, use_weights=True):
    npix = 12 * nside * nside
    oop  = float(npix)/(4*np.pi) # 1/(solid angle of a pixel) 
    marr = np.eye(npix) * oop
    Tarr = marr * 0
    Qarr = marr
    Uarr = marr * 0

    alms = [hp.map2alm([T, Q, U], use_weights=use_weights, iter=100) for T, Q, U in zip(Tarr, Qarr, Uarr)]
    alms = np.array(alms)

    wlm = np.conj(alms[:,1,:])
    xlm = np.conj(alms[:,2,:])

    return wlm, xlm
    

def gen_Pl_nk(nside, lmax=None, ylm=None, use_weights=True):
    if lmax is None:
        lmax = 3*nside - 1

    prename = './precomputed/Pl_nk_nside{}_lmax{}_compressed.npz'.format(nside, lmax)
    if os.path.isfile(prename):
        print ('Loading precomputed Pl arrays ...')
        data = np.load(prename)
        return data['pl']

    if (ylm==None):
        ylm = gen_ylm(nside, use_weights=use_weights)

    ylm = np.array(ylm)
    pl = []

    for l in range(lmax+1):
        print(ylm.shape)
        y0  = ylm[:,l]
        y0c = np.conj(y0)
        pltmp = np.outer(y0, y0c)
        if (l > 0):
            m   = np.arange(l)+1
            idx = [hp.Alm.getidx(lmax, l, mm) for mm in m]
            ym  = ylm[:, idx]
            ymc = np.conj(ym)
            pltmp += np.einsum('lm,km->lk', ym, ymc) \
                   + np.einsum('lm,km->lk', ymc, ym)
        pl.append(pltmp.real)

    try:
        os.mkdir('./precomputed')
    except :
        pass

    np.savez_compressed(prename, pl=pl) 
        
    return np.array(pl)


def gen_Wls_nk(nside, lmax=None, use_weights=True):
    if lmax is None:
        lmax = 3 * nside - 1

    prename = './precomputed/Wls_nk_nside{}_lmax{}_compressed.npz'.format(nside, lmax)
    if os.path.isfile(prename):
        print ('Loading precomputed Wl arrays ... ')
        data = np.load(prename)
        return [data['Wl11'], data['Wl22'], data['Wl01'], data['Wl02'], data['Wl12'], data['Wl21']]

    wlm, xlm = gen_wxlm(nside, use_weights=use_weights)
    Wl11 = []
    Wl12 = []
    Wl21 = []
    Wl22 = []
    Wl01 = []
    Wl02 = []

    for l in range(lmax+1):
        w0 = wlm[:,l]
        x0 = xlm[:,l]
        w0c = np.conj(w0)
        x0c = np.conj(x0)
        Wl11tmp = np.outer(w0, w0c)
        Wl12tmp = np.outer(-w0, x0c)
        Wl21tmp = np.outer(-x0, w0c)
        Wl22tmp = np.outer(x0, x0c)
        if (l > 0):
            m = np.arange(l) + 1
            idx = [hp.Alm.getidx(lmax, l, mm) for mm in m]
            wm = wlm[:, idx]
            xm = xlm[:, idx]
            wmc = np.conj(wm)
            xmc = np.conj(xm)
            Wl11tmp += np.einsum('lm,km->lk',  wm, wmc) \
                     + np.einsum('lm,km->lk',  wmc, wm)
            Wl12tmp += np.einsum('lm,km->lk', -wm, xmc) \
                     + np.einsum('lm,km->lk', -wmc, xm)
            Wl21tmp += np.einsum('lm,km->lk', -xm, wmc) \
                     + np.einsum('lm,km->lk', -xmc, wm)
            Wl22tmp += np.einsum('lm,km->lk',  xm, xmc) \
                     + np.einsum('lm,km->lk',  xmc, xm)

        m0 = np.zeros(Wl11tmp.shape)
        Wl11.append(Wl11tmp)
        Wl12.append(Wl12tmp)
        Wl21.append(Wl21tmp)
        Wl22.append(Wl22tmp)
        Wl01.append(m0)
        Wl02.append(m0)
     
    Wl11 = np.array(Wl11)
    Wl12 = np.array(Wl12)
    Wl21 = np.array(Wl21)
    Wl22 = np.array(Wl22)
    Wl01 = np.array(Wl01)
    Wl02 = np.array(Wl02)

    try:
        os.mkdir('./precomputed')
    except:
        pass

    np.savez_compressed(prename, Wl11=Wl11, Wl22=Wl22, Wl01=Wl01, Wl02=Wl02, Wl12=Wl12, Wl21=Wl21)


    return [Wl11, Wl22, Wl01, Wl02, Wl12, Wl12]


def getcov_nk(cls, nside=4, pls=None, lmax=None, isDl=False, use_weights=True):
    if (lmax==None):
        lmax = nside * 3 - 1

    if (isDl):
        cls = dl2cl(cls)
    else:
        cls = cls.copy()

    if len(cls.shape) == 2:
        cls = cls[0]

    cls = cls[:lmax+1]

    if (type(pls)==type(None)):
        pls = gen_Pl_nk(nside, use_weights=use_weights)

    cov = np.einsum('l,lij->ij', cls, pls)

    return cov


def getcov_nk_pol(cls, nside=4, pls=None, wls=None, lmax=None, isDl=False, use_weights=True):
    if (lmax == None):
        lmax = 3*nside - 1

    if pls is None:
        pls = gen_Pl_nk(nside, lmax, use_weights=use_weights)

    if wls is None:
        Wl11, Wl22, Wl01, Wl02, Wl12, Wl21 = gen_Wls_nk(nside, lmax, use_weights=use_weights)
    else:
        Wl11, Wl22, Wl01, Wl02, Wl12, Wl21 = wls

    if (isDl):
        cls = dl2cl(cls)
    else:
        cls = cls.copy()

    cls = cls[:,:lmax+1]

    clTT, clEE, clBB, clTE = cls
    ell = np.arange(len(clTT))

    covTT = np.einsum('l,lij->ij', clTT, pls)

    covQQ = np.einsum('l,lij->ij', clEE, Wl11) + np.einsum('l,lij->ij', clBB, Wl22)
    covUU = np.einsum('l,lij->ij', clEE, Wl22) + np.einsum('l,lij->ij', clBB, Wl11)
    covQU = np.einsum('l,lij->ij', clEE, Wl12) + np.einsum('l,lij->ij', clBB, -1*np.conj(Wl21))
    covUQ = np.einsum('l,lij->ij', clEE, Wl21) + np.einsum('l,lij->ij', clBB, -1*np.conj(Wl12))
    covTQ = np.einsum('l,lij->ij', clTE, Wl01)
    covTU = np.einsum('l,lij->ij', clTE, Wl02)
    cov00 = np.zeros(np.shape(covTT))

    cov = [[covTT, covTQ, covTU], [covTQ.T, covQQ, covQU.T], [covTU.T, covUQ.T, covUU]]
    cov = np.concatenate(np.concatenate(cov, 1),1)

    return cov.real
 

