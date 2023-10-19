import os
import time

import numpy as np
import healpy as hp

from scipy.special import legendre, lpmv, factorial
from tqdm import tqdm

from .utils import dl2cl


def sign0(x):
    s = np.sign(x)
    if s == 0:
        s = 1
    return s


def crosss(v1, v2):
    c = np.cross(v1, v2)
    c /= np.linalg.norm(c) * sign0(c[2])
    return c


def correct_cos(cos_in):
    if cos_in > 1.0: 
        cos_out = 1.0 
    elif cos_in < -1.0:
        cos_out = -1.0
    else:
        cos_out = cos_in

    return cos_out


def get_cosbeta(v1, v2):
    cosbeta = np.dot(v1, v2)/np.linalg.norm(v1)/np.linalg.norm(v2)
    cosbeta = correct_cos(cosbeta)
    return cosbeta


def get_cos_sin_beta(v1, v2):
    cosbeta = get_cosbeta(v1, v2)
    sinbeta = np.sqrt(1.0-cosbeta**2)

    if sinbeta == 0:
        sinbeta += 1e-5
        cosbeta = (1-sinbeta**2)**0.5 * np.sign(cosbeta) 

    return cosbeta, sinbeta


def get_cos_sin_2psis_old(v1, v2): # outdated
    z = (0, 0, 1)
    v1 = v1.copy()
    v2 = v2.copy()

    if (v1 == v2).all():
        v2[2] += 1e-9

    c1 = np.cross(z, v1)
    c2 = np.cross(z, v2)
    c12 = np.cross(v1, v2)
    c1 /= np.linalg.norm(c1)
    c2 /= np.linalg.norm(c2)
    c12 /= np.linalg.norm(c12)

    cos2psi1 = (2*np.dot(c1, c12)**2-1)
    cos2psi1 = correct_cos(cos2psi1)
    sin2psi1 = (1 - cos2psi1**2)**0.5

    cos2psi2 = (2*np.dot(c2, c12)**2-1)
    cos2psi2 = correct_cos(cos2psi2)
    sin2psi2 = (1 - cos2psi2**2)**0.5

    return cos2psi1, sin2psi1, cos2psi2, sin2psi2
    

def get_cos_sin_2psis(v1, v2):
    z = (0, 0, 1)
    v1 = v1.copy()
    v2 = v2.copy()

    if (np.linalg.norm(np.cross(v1, v2)) < 1e-15):
        return 1, 0, 1, 0
        #v2[2] += 1e-15
        #v2 /= np.linalg.norm(v2)

    phi1 = crosss(z, v1)
    phi2 = crosss(z, v2)

    c12 = crosss(v1, v2)

    phip1 = crosss(c12, v1)
    phip2 = crosss(c12, v2) 
    psi1 = np.arccos(get_cosbeta(phi1, phip1))
    psi2 = np.arccos(get_cosbeta(phi2, phip2))

    cos2psi1 = np.cos(2*psi1)
    sin2psi1 = np.sin(2*psi1) 

    cos2psi2 = np.cos(2*psi2)
    sin2psi2 = np.sin(2*psi2)

    return cos2psi1, sin2psi1, cos2psi2, sin2psi2
    

def lpn_dm(m, n, x):
    if n < 0:
        return np.zeros(x.shape) 

    lp = legendre(n)
    f = lp.deriv(m)
    return f(x)


def gen_Pl_ana(nside=4, lmax=None, prename=None):
    if lmax is None:
        lmax = 3*nside - 1

    print ('Computing Pl arrays ...')
    npix = 12 * nside * nside
    pixarr = np.arange(npix)

    ## npix array -> vectors
    vecarr = np.array(hp.pix2vec(nside, pixarr)).T

    ## two vectors' pairs -> Pls
    cosbeta = np.zeros((npix, npix))
    for i in range(npix):
        for j in range(npix):
            cosbeta[i][j] = get_cosbeta(vecarr[i], vecarr[j])

    ell = np.arange(lmax+1)
    legs = [legendre(l) for l in ell]
    pl = np.array([fnc(cosbeta) for fnc in legs])
    for l in ell:
        pl[l] *= (2*l+1)/4/np.pi 

    if isinstance(prename, str):
        try:
            os.mkdir('./precomputed')
        except:
            print ('The precomputed directory already exists.')

        np.savez_compressed(prename, pl=pl) 

    return pl


def gen_Flm_ana(m, nside=4, lmax=None): # outdated
    if (lmax==None):
        lmax = nside*3-1

    npix = 12 * nside * nside
    pixarr = np.arange(npix)

    ## npix array -> vectors
    vecarr = np.array(hp.pix2vec(nside, pixarr)).T

    ## two vectors' pairs -> Pls
    cosbeta = np.zeros((npix, npix))
    sinbeta = np.zeros((npix, npix))

    for i in range(npix):
        for j in range(npix):
            cosbeta[i][j] = get_cosbeta(vecarr[i], vecarr[j])
            if (cosbeta[i][j] > 1.0):
                cosbeta[i][j] = 1.0
                sinbeta[i][j] = 0.0
            elif (cosbeta[i][j] < -1.0):
                cosbeta[i][j] = -1.0
                sinbeta[i][j] = 0.0
            else:
                sinbeta[i][j] = np.sqrt(1.0-cosbeta[i][j]**2)

            if sinbeta[i][j] == 0.0:
                sinbeta[i][j] += 1e-9
                cosbeta[i][j] = np.sqrt(1.0-sinbeta[i][j]**2)

    ell = np.arange(lmax+1)
    Plm = np.array([lpmv(m, l, cosbeta) for l in ell])
    Pl_1m = np.array([lpmv(m, l-1, cosbeta) for l in ell])
    
    Nlm = 2*np.sqrt( factorial(ell-2) * factorial(ell-m) / factorial(ell+2) / factorial(ell+m) )
    F1lm = []
    F2lm = []

    for l in ell:
        F1lm.append( Nlm[l] * ( -((l-m**2)/(sinbeta**2) + 0.5*l*(l-1))*Plm[l] + (l+m)*cosbeta/(sinbeta**2)*Pl_1m[l] ) )
        F2lm.append( Nlm[l] * m / (sinbeta**2) * ( -(l-1)*cosbeta*Plm[l] + (l+m)*Pl_1m[l] )  )

    F1lm = np.array(F1lm)
    F2lm = np.array(F2lm)

    return F1lm, F2lm


def gen_Wls_ana(nside=4, lmax=None, prename=None):
    if (lmax==None):
        lmax = nside*3-1

    print ('Computing Wl arrays ...')
    npix = 12 * nside * nside
    pixarr = np.arange(npix)

    ## npix array -> vectors
    vecarr = np.array(hp.pix2vec(nside, pixarr)).T

    ## two vectors' pairs -> Pls
    cosbeta = np.zeros((npix, npix))
    sinbeta = np.zeros((npix, npix))
    cos2psi1 = np.zeros((npix, npix))
    sin2psi1 = np.zeros((npix, npix))
    cos2psi2 = np.zeros((npix, npix))
    sin2psi2 = np.zeros((npix, npix))

    ## this double loop can be replaced with array calculations
    for i in tqdm(range(npix)):
        for j in range(npix):
            cosbeta[i][j], sinbeta[i][j] = get_cos_sin_beta(vecarr[i], vecarr[j])
            cos2psi1[i][j], sin2psi1[i][j], cos2psi2[i][j], sin2psi2[i][j] = get_cos_sin_2psis(vecarr[i], vecarr[j])
     
    ell = np.arange(lmax+1)

    #Pl2 = np.array([lpn_dm(2, l, cosbeta) for l in ell]) * sinbeta**2
    #Pl_12 = np.array([lpn_dm(2, l-1, cosbeta) for l in ell]) * sinbeta**2

    #Pl0 = np.array([lpn_dm(0, l, cosbeta) for l in ell])
    #Pl_10 = np.array([lpn_dm(0, l-1, cosbeta) for l in ell]) 

    Pl2 = np.array([lpmv(2, l, cosbeta) for l in ell])
    Pl_12 = np.array([lpmv(2, l-1, cosbeta) for l in ell])

    Pl0 = np.array([lpmv(0, l, cosbeta) for l in ell])
    Pl_10 = np.array([lpmv(0, l-1, cosbeta) for l in ell])
    
    Nl2 = 2*np.sqrt( factorial(ell-2) * factorial(ell-2) / factorial(ell+2) / factorial(ell+2) )
    Nl0 = 2*np.sqrt( factorial(ell-2) * factorial(ell-0) / factorial(ell+2) / factorial(ell+0) )

    Wl11 = [] # sum_m [X^*_1lm(1) X_1lm(2)]
    Wl22 = [] # sum_m [X^*_2lm(1) X_2lm(2)]
    Wl12 = [] # i * sum_m [X^*_1lm(1) X_2lm(2)]
    Wl21 = [] # -i * sum_m [X^*_2lm(1) X_1lm(2)]
    Wl01 = [] # sum_m [0Y^*_1lm(1) X_1lm(2)]
    Wl02 = [] # -i * sum_m [0Y^*_1lm(1) X_2lm(2)]

    for l in ell:
        norm_l = (2*l+1) / 4 / np.pi
        F1l2 = Nl2[l] * ( -((l-2**2)/(sinbeta**2) + 0.5*l*(l-1))*Pl2[l] + (l+2)*cosbeta/(sinbeta**2)*Pl_12[l] ) 
        F2l2 = Nl2[l] * 2 / (sinbeta**2) * ( -(l-1)*cosbeta*Pl2[l] + (l+2)*Pl_12[l] )  
        F1l0 = Nl0[l] * ( -((l-0**2)/(sinbeta**2) + 0.5*l*(l-1))*Pl0[l] + (l+0)*cosbeta/(sinbeta**2)*Pl_10[l] ) 
        F2l0 = 0 

        Wl11.append(norm_l * (F1l2 * cos2psi1 * cos2psi2 - F2l2 * sin2psi1 * sin2psi2))
        Wl22.append(norm_l * (F1l2 * sin2psi1 * sin2psi2 - F2l2 * cos2psi1 * cos2psi2))
        #Wl12.append(norm_l * 1j * (F1l2 * sin2psi1 * cos2psi2 + F2l2 * cos2psi1 * sin2psi2))
        #Wl21.append(norm_l * -1j * (F1l2 * cos2psi1 * sin2psi2 + F2l2 * sin2psi1 * cos2psi2))
        Wl12.append(norm_l * 1j * (F1l2 * sin2psi1 * cos2psi2 + F2l2 * cos2psi1 * sin2psi2))
        Wl21.append(norm_l * -1j * (F1l2 * cos2psi1 * sin2psi2 + F2l2 * sin2psi1 * cos2psi2))
        Wl01.append(norm_l * F1l0 * cos2psi2)
        Wl02.append(norm_l * -1j * F1l0 * sin2psi2)

    Wl11 = np.array(Wl11)
    Wl22 = np.array(Wl22)
    Wl12 = np.array(Wl12)
    Wl21 = np.array(Wl21)
    Wl01 = np.array(Wl01)
    Wl02 = np.array(Wl02)

    if isinstance(prename, str):
        try:
            os.mkdir('./precomputed')
        except:
            print ('The precomputed directory already exists.')

        np.savez_compressed(prename, Wl11=Wl11, Wl22=Wl22, Wl01=Wl01, Wl02=Wl02, Wl12=Wl12, Wl21=Wl21)

    return [Wl11, Wl22, Wl01, Wl02, Wl12, Wl21]


def getcov_ana(cls, nside=4, pls=None, wls=None, lmax=None, isDl=False, pol=False):
    if pol:
        cov = getcov_ana_pol(cls, nside, pls, wls, lmax, isDl)
        return cov

    if (lmax==None):
        lmax = 3*nside-1

    if (isDl):
        cls = dl2cl(cls)
    else:
        cls = cls.copy()

    if len(cls.shape) == 2:
        cls = cls[0]

    cls = cls[:lmax+1]
    if (type(pls)==type(None)):
        pls = gen_Pl_ana(nside, lmax)

    ## l sum with (2l+1)Cl/4pi -> covariance
    ell = np.arange(len(cls))
    bls = (2.*ell+1)/4/np.pi * cls
    cov = np.einsum('l,lij->ij', bls, pls)

    return cov


def getcov_ana_pol(cls, nside=4, pls=None, wls=None, lmax=None, isDl=False):
    if (lmax==None):
        lmax = 3*nside-1

    ## for the details of the computation, 
    ## see chapter 2 of arXiv:astro-ph/9806122

    if isinstance(pls, str):
        prename = pls #'./precomputed/Pl_nside{}_lmax{}_compressed.npz'.format(nside, lmax)
        if os.path.isfile(prename):
            print ('Loading precomputed Pl arrays ...')
            data = np.load(prename)
            pls = data['pl'] 
        else:
            pls = gen_Pl_ana(nside, lmax, prename=prename)
    elif hasattr(pls, '__iter__'):
        pls = pls
    elif pls is None:
        pls = gen_Pl_ana(nside, lmax)
    else:
        exit(-1)

    if isinstance(wls, str):
        prename = wls #'./precomputed/Wls_nside{}_lmax{}_compressed.npz'.format(nside, lmax)
        if os.path.isfile(prename):
            print ('Loading precomputed Wl arrays ... ')
            data = np.load(prename)
            wls = [data['Wl11'], data['Wl22'], data['Wl01'], data['Wl02'], data['Wl12'], data['Wl21']]
        else:
            wls = gen_Wls_ana(nside, lmax, prename=prename)
    elif hasattr(wls, '__iter__'): 
        wls = wls
    elif wls is None:
        wls = gen_Wls_ana(nside, lmax)
    else:
        exit(-1)
        
    Wl11, Wl22, Wl01, Wl02, Wl12, Wl21 = wls 

    ## l sum Cl -> covariance
    if (isDl):
        cls = dl2cl(cls)
    else:
        cls = cls.copy()

    cls = cls[:,:lmax+1]

    clTT, clEE, clBB, clTE = cls
    ell = np.arange(len(clTT))

    covTT = np.einsum('l,lij->ij', clTT, pls)

    covQQ = np.einsum('l,lij->ij', clEE, Wl11) + np.einsum('l,lij->ij', clBB, Wl22)
    covUU = np.einsum('l,lij->ij', clBB, Wl11) + np.einsum('l,lij->ij', clEE, Wl22)
    #covQU = np.einsum('l,lij->ij', clEE, 1j*np.conj(Wl12)) + np.einsum('l,lij->ij', clBB, -1j*np.conj(Wl21))
    #covUQ = np.einsum('l,lij->ij', clBB, -1j*np.conj(Wl12)) + np.einsum('l,lij->ij', clEE, -1j*np.conj(Wl21))
    covQU = np.einsum('l,lij->ij', clEE, np.imag(Wl12)) + np.einsum('l,lij->ij', clBB, -1*np.imag(Wl21))
    covUQ = np.einsum('l,lij->ij', clBB, -1*np.imag(Wl12)) + np.einsum('l,lij->ij', clEE, -1*np.imag(Wl21))
    covTQ = np.einsum('l,lij->ij', clTE, Wl01)
    covTU = 1j*np.einsum('l,lij->ij', clTE, Wl02)
    cov00 = np.zeros(np.shape(covTT))

    ## original
    #cov = [[covTT, covTQ, covTU], [covTQ.T, covQQ, covQU.T], [covTU.T, covUQ.T, covUU]]
    ## corrected
    cov = [[covTT, covTQ, covTU], [covTQ.T, covQQ, covUQ], [covTU.T, covUQ.T, covUU]]
    cov = np.concatenate(np.concatenate(cov, 1),1)

    return cov.real
 

def generate_plfl(nside=4, lmax=None):
    if lmax is None:
        lmax = 3*nside - 1
    else:
        if lmax > nside*3-1:
            print('lmax is greater than (3*nside-1)')
            lmax = 3*nside - 1

    t0 = time.time()
    pls          = gen_Pl_ana(nside, lmax)
    f1l2s, f2l2s = gen_Flm_ana(2, nside, lmax) 
    f1l0s, f2l0s = gen_Flm_ana(0, nside, lmax) 

    print('time for computing plfl: {} s'.format(time.time() - t0))

    if not(os.path.isdir('plfl')):
        os.mkdir('plfl')

    fname = 'plfl/plfl_nside{}_lmax{}.npz'.format(nside, lmax)

    np.savez(fname,
             pls=pls, f1l0s=f1l0s, f2l0s=f2l0s, f1l2s=f1l2s, f2l2s=f2l2s)

    print('The pl and fl arrays are written in {}.'.format(fname))


def generate_plwl(nside=4, lmax=None):
    if lmax is None:
        lmax = 3*nside - 1
    else:
        if lmax > nside*3-1:
            print('lmax is greater than (3*nside-1)')
            lmax = 3*nside - 1

    t0 = time.time()
    pls          = gen_Pl_ana(nside, lmax)
    f1l2s, f2l2s = gen_Flm_ana(2, nside, lmax) 
    f1l0s, f2l0s = gen_Flm_ana(0, nside, lmax) 

    print('time for computing plfl: {} s'.format(time.time() - t0))

    if not(os.path.isdir('plfl')):
        os.mkdir('plfl')

    fname = 'plfl/plfl_nside{}_lmax{}.npz'.format(nside, lmax)

    np.savez(fname,
             pls=pls, f1l0s=f1l0s, f2l0s=f2l0s, f1l2s=f1l2s, f2l2s=f2l2s)

    print('The pl and fl arrays are written in {}.'.format(fname))


