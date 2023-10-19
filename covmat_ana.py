import os
import time

import numpy as np
import healpy as hp

from scipy.special import legendre, lpmv, factorial, binom
from tqdm import tqdm
from .utils import dl2cl
from gbpipe.utils import set_logger

LVLV = 'DEBUG'


def sign0(x):
    s = np.sign(x)
    if hasattr(s, '__iter__'):
        s[s==0] = 1
    else:
        if s == 0: 
            s = 1

    return s


def crosss(v1, v2):
    if len(np.shape(v1)) > 1 or len(np.shape(v2)) > 1:
        return crosss_arr(v1, v2)

    c = np.cross(v1, v2)
    cn = np.linalg.norm(c)
    if cn < 1e-5: 
        cn = 1.0 
    c /= cn * sign0(c[2])
    return c


def crosss_arr(v1, v2):
    c = np.cross(v1, v2)
    cn = np.linalg.norm(c, axis=2)
    cn[cn<1e-5] = 1.0 
    cs = sign0(c[...,2])
    c[...,0] = c[...,0] / cn / cs
    c[...,1] = c[...,1] / cn / cs
    c[...,2] = c[...,2] / cn / cs


    return c


def correct_cos(cos_in):
    if cos_in > 1.0: 
        cos_out = 1.0 - 1e-5
    elif cos_in < -1.0:
        cos_out = -1.0 + 1e-5
    else:
        cos_out = cos_in

    return cos_out


def get_cosbeta(v1, v2):
    if len(np.shape(v1)) > 1 or len(np.shape(v2)) > 1:
        return get_cosbeta_arr(v1, v2)

    cosbeta = np.dot(v1, v2)/np.linalg.norm(v1)/np.linalg.norm(v2)
    cosbeta = correct_cos(cosbeta)
    
    return cosbeta


def get_cosbeta_arr(v1, v2):
    n1 = np.linalg.norm(v1, axis=-1) 
    n2 = np.linalg.norm(v2, axis=-1) 
    d12 = np.sum(v1 * v2, axis=-1)
    cosbeta = d12 / n1 / n2
    cosbeta[cosbeta>1.0] = 1.0 - 1e-5
    cosbeta[cosbeta<-1.0] = -1.0 + 1e-5

    return cosbeta


def get_cos_sin_beta(v1, v2):
    cosbeta = get_cosbeta(v1, v2)
    sinbeta = np.sqrt(1.0-cosbeta**2)

    pert = 1e-5

    if hasattr(sinbeta, '__iter__'):
        #pass
        sinbeta[sinbeta<pert] = pert
        cosbeta = np.sqrt(1-sinbeta**2) * np.sign(cosbeta)
    else:
        if sinbeta < pert:
            #pass
            sinbeta = pert
            cosbeta = np.sqrt(1-sinbeta**2) * np.sign(cosbeta) 

    return cosbeta, sinbeta


def get_cos_sin_2psis(v1, v2):
    if (len(np.shape(v1)) > 1 or len(np.shape(v2)) > 1):
        return get_cos_sin_2psis_arr(v1, v2)

    z = (0, 0, 1)
    v1 = v1.copy()
    v2 = v2.copy()

    phi1 = crosss(z, v1)
    phi2 = crosss(z, v2)

    c12 = np.cross(v1, v2)
    if (np.linalg.norm(c12) < 1e-15):
        c12 = phi2
        #pass
        #return 1, 0, 1, 0

    phip1 = crosss(c12, v1)
    phip2 = crosss(c12, v2) 

    psi1 = np.arccos(get_cosbeta(phi1, phip1))
    psi2 = np.arccos(get_cosbeta(phi2, phip2))
    #psi1 = np.arccos(get_cosbeta(phi1, c12))
    #psi2 = np.arccos(get_cosbeta(phi2, c12))

    cos2psi1 = np.cos(2*psi1)
    sin2psi1 = np.sin(2*psi1) 

    cos2psi2 = np.cos(2*psi2)
    sin2psi2 = np.sin(2*psi2)

    return cos2psi1, sin2psi1, cos2psi2, sin2psi2
    

def get_cos_sin_2psis_arr(v1, v2):
    z = (0, 0, 1)

    for l in np.shape(v1)[:-1]:
        z = [z] * l

    z = np.array(z)
    phi1 = crosss(z, v1)
    phi2 = crosss(z, v2)

    c12 = crosss(v1, v2)

    where0 = np.where(np.linalg.norm(c12, axis=2) == 0)
    c12[where0] = phi1[where0]

    # is this a bug?
    phip1 = crosss(c12, v1) # thetap1
    phip2 = crosss(c12, v2) # thetap2

    psi1 = np.arccos(get_cosbeta(phi1, phip1)) 
    psi2 = np.arccos(get_cosbeta(phi2, phip2))
    #psi1 = np.arccos(get_cosbeta(phi1, c12)) 
    #psi2 = np.arccos(get_cosbeta(phi2, c12))

    cos2psi1 = np.cos(2*psi1)
    sin2psi1 = np.sin(2*psi1) 

    cos2psi2 = np.cos(2*psi2)
    sin2psi2 = np.sin(2*psi2)
    
    #cos2psi1[where0] = cos2psi2[where0] = 1
    #sin2psi1[where0] = sin2psi2[where0] = 0

    return cos2psi1, sin2psi1, cos2psi2, sin2psi2


def lpn_dm(m, n, x):
    if n < 0:
        return np.zeros(x.shape) 

    lp = legendre(n)
    f = lp.deriv(m)

    return f(x)


def lpn_d2_binom(m, n, x): # m should be 2
    k = np.arange(n+1) + m 

    res = 0 
    for kk in k: 
        b1 = binom(n, kk) 
        b2 = binom((n+kk-1)*0.5, n) 
        p1 = 2**n 
        p2 = kk*(kk-1) 
        p3 = x ** (kk-2) 
        #res += 2**n * kk*(kk-1) * x**(kk-2) * b1 * b2 
        tmp = p1 * p2 * p3 * b1 * b2
        res += tmp

    return res 


def gen_Pl_ana_loop(nside=4, lmax=None, prename=None):
    if lmax is None:
        lmax = 3*nside - 1

    if isinstance(prename, str): 
        if os.path.isfile(prename):
            print ('Loading precomputed Pl arrays ...')
            data = np.load(prename)
            pls = data['pl'] 
            return pls
        
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
            if not os.path.isdir('./precomputed'):
                os.mkdir('./precomputed')
            np.savez_compressed(prename, pl=pl) 
        except:
            print ('invalid prename is given.')

    return pl


def gen_Pl_ana(nside=4, lmax=None, prename=None):
    if lmax is None:
        lmax = 3*nside - 1

    if isinstance(prename, str): 
        if os.path.isfile(prename):
            print ('Loading precomputed Pl arrays ...')
            data = np.load(prename)
            pls = data['pl'] 
            return pls
        
    print ('Computing Pl arrays ...')
    npix = 12 * nside * nside
    pixarr = np.arange(npix)

    ## npix array -> vectors
    vecarr = np.array(hp.pix2vec(nside, pixarr)).T

    ## two vectors' pairs -> Pls
    vgrid1 = np.array([vecarr] * len(vecarr))
    vgrid2 = np.transpose(vgrid1, (1,0,2))
    cosbeta = get_cosbeta(vgrid1, vgrid2)

    ell = np.arange(lmax+1)
    legs = [legendre(l) for l in ell]
    pl = np.array([fnc(cosbeta) for fnc in legs])
    for l in ell:
        pl[l] *= (2*l+1)/4/np.pi 

    if isinstance(prename, str):
        try:
            if not os.path.isdir('./precomputed'):
                os.mkdir('./precomputed')
            np.savez_compressed(prename, pl=pl) 
        except:
            print ('invalid prename is given.')

    return pl


def gen_Wls_ana_loop(nside=4, lmax=None, prename=None):
    if (lmax==None):
        lmax = nside*3-1

    if isinstance(prename, str):
        if os.path.isfile(prename):
            print ('Loading precomputed Wl arrays ... ')
            data = np.load(prename)
            wls = [data['Wl11'], data['Wl22'], data['Wl01'], data['Wl02'], data['Wl12'], data['Wl21']]
            return wls        

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

    Pl2 = np.array([lpn_d2_binom(2, l, cosbeta) for l in ell])
    Pl_12 = np.array([lpn_d2_binom(2, l-1, cosbeta) for l in ell])

    Pl0 = np.array([lpn_dm(0, l, cosbeta) for l in ell])
    Pl_10 = np.array([lpn_dm(0, l-1, cosbeta) for l in ell])

    #Pl2 = np.array([lpmv(2, l, cosbeta) for l in ell])
    #Pl_12 = np.array([lpmv(2, l-1, cosbeta) for l in ell])

    #Pl0 = np.array([lpmv(0, l, cosbeta) for l in ell])
    #Pl_10 = np.array([lpmv(0, l-1, cosbeta) for l in ell])
    
    Nl2 = 2*np.sqrt( factorial(ell-2) * factorial(ell-2) / factorial(ell+2) / factorial(ell+2) )
    Nl0 = 2*np.sqrt( factorial(ell-2) * factorial(ell-0) / factorial(ell+2) / factorial(ell+0) )

    Wl11 = [] # sum_m [X^*_1lm(1) X_1lm(2)]
    Wl22 = [] # sum_m [X^*_2lm(1) X_2lm(2)]
    Wl12 = [] # i * sum_m [X^*_1lm(1) X_2lm(2)]
    Wl21 = [] # -i * sum_m [X^*_2lm(1) X_1lm(2)]
    Wl01 = [] # sum_m [0Y^*_1lm(1) X_1lm(2)]
    Wl02 = [] # -i * sum_m [0Y^*_1lm(1) X_2lm(2)]

    for l in ell:
        norm_l = (2*l + 1) / 4 / np.pi
        #F1l2 = Nl2[l] * ( -((l-2**2)/(sinbeta**2) + 0.5*l*(l-1))*Pl2[l] + (l+2)*cosbeta/(sinbeta**2)*Pl_12[l] ) 
        #F2l2 = Nl2[l] * 2 / (sinbeta**2) * ( -(l-1)*cosbeta*Pl2[l] + (l+2)*Pl_12[l] )  
        #F1l0 = Nl0[l] * ( -((l-0**2)/(sinbeta**2) + 0.5*l*(l-1))*Pl0[l] + (l+0)*cosbeta/(sinbeta**2)*Pl_10[l] ) 
        F1l2 = Nl2[l] * ( -((l-2**2) + 0.5*l*(l-1)*(sinbeta**2))*Pl2[l] + (l+2)*cosbeta*Pl_12[l] ) 
        F2l2 = Nl2[l] * 2 * ( -(l-1)*cosbeta*Pl2[l] + (l+2)*Pl_12[l] )  
        F1l0 = Nl0[l] * ( -((l-0**2) + 0.5*l*(l-1)*(sinbeta**2))*Pl0[l] + (l+0)*cosbeta*Pl_10[l] )/sinbeta**2 
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
            if not os.path.isdir('./precomputed'):
                os.mkdir('./precomputed')
            np.savez_compressed(prename, Wl11=Wl11, Wl22=Wl22, Wl01=Wl01, Wl02=Wl02, Wl12=Wl12, Wl21=Wl21)
        except:
            print ('The prename is invalid.')

    return [Wl11, Wl22, Wl01, Wl02, Wl12, Wl21]


def gen_Wls_ana(nside=4, lmax=None, prename=None): 
    logger = set_logger(level=LVLV)
    if (lmax==None):
        lmax = nside*3-1

    if isinstance(prename, str):
        if os.path.isfile(prename):
            print ('Loading precomputed Wl arrays ... ')
            data = np.load(prename)
            wls = [data['Wl11'], data['Wl22'], data['Wl01'], data['Wl02'], data['Wl12'], data['Wl21']]
            return wls        

    print ('Computing Wl arrays ...')
    npix = 12 * nside * nside
    pixarr = np.arange(npix)

    ## npix array -> vectors
    vecarr = np.array(hp.pix2vec(nside, pixarr)).T

    ## two vectors' pairs -> Pls
    logger.debug('starting calculations')
    t0 = time.time()

    vgrid1 = np.array([vecarr] * len(vecarr)).transpose((1, 0, 2))
    vgrid2 = np.transpose(vgrid1, (1,0,2))

    cosbeta, sinbeta = get_cos_sin_beta(vgrid1, vgrid2)
    cos2psi1, sin2psi1, cos2psi2, sin2psi2 = get_cos_sin_2psis(vgrid1, vgrid2)

    logger.debug(f'time for angle calculations: {time.time()-t0}')

    ell = np.arange(lmax+1)

    t0 = time.time()

    Pl2 = np.array([lpn_dm(2, l, cosbeta) for l in ell])
    Pl_12 = np.array([lpn_dm(2, l-1, cosbeta) for l in ell])

    Pl0 = np.array([lpn_dm(0, l, cosbeta) for l in ell])
    Pl_10 = np.array([lpn_dm(0, l-1, cosbeta) for l in ell])

    #Pl2 = np.array([lpn_d2_binom(2, l, cosbeta) for l in ell])
    #Pl_12 = np.array([lpn_d2_binom(2, l-1, cosbeta) for l in ell])

    #Pl0 = np.array([lpn_d2_binom(0, l, cosbeta) for l in ell])
    #Pl_10 = np.array([lpn_d2_binom(0, l-1, cosbeta) for l in ell])

    # sinbeta=0 components using new formulae
    where0 = np.where(np.abs(sinbeta) < 1e-5)

    logger.debug(f'time for legendre calculations: {time.time()-t0}')
    
    t0 = time.time()
    Nl2 = 2*np.sqrt( factorial(ell-2) * factorial(ell-2) / factorial(ell+2) / factorial(ell+2) )
    Nl0 = 2*np.sqrt( factorial(ell-2) * factorial(ell-0) / factorial(ell+2) / factorial(ell+0) )
    #Nl2 = 2/( (ell+2)*(ell+1)*ell*(ell-1) )
    #Nl0 = 2/np.sqrt( (ell+2)*(ell+1)*ell*(ell-1) )
    #Nl2[0] = 0
    #Nl2[1] = 0
    #Nl0[0] = 0
    #Nl0[1] = 0

    logger.debug(f'time for normalization calculations: {time.time()-t0}')

    Wl11 = [] # sum_m [X^*_1lm(1) X_1lm(2)]
    Wl22 = [] # sum_m [X^*_2lm(1) X_2lm(2)]
    Wl12 = [] # i * sum_m [X^*_1lm(1) X_2lm(2)]
    Wl21 = [] # -i * sum_m [X^*_2lm(1) X_1lm(2)]
    Wl01 = [] # sum_m [0Y^*_1lm(1) X_1lm(2)]
    Wl02 = [] # -i * sum_m [0Y^*_1lm(1) X_2lm(2)]

    t0 = time.time()
    for l in ell:
        norm_l = (2*l+1) / 4 / np.pi
        # old formulae
        #F1l2 = Nl2[l] * ( -((l-2**2)/(sinbeta**2) + 0.5*l*(l-1))*Pl2[l] + (l+2)*cosbeta/(sinbeta**2)*Pl_12[l] ) 
        #F2l2 = Nl2[l] * 2 / (sinbeta**2) * ( -(l-1)*cosbeta*Pl2[l] + (l+2)*Pl_12[l] )  
        #F1l0 = Nl0[l] * ( -((l-0**2)/(sinbeta**2) + 0.5*l*(l-1))*Pl0[l] + (l+0)*cosbeta/(sinbeta**2)*Pl_10[l] ) 

        # new formulae 
        F1l2 = Nl2[l] * ( -((l-2**2) + 0.5*l*(l-1)*(sinbeta**2))*Pl2[l] + (l+2)*cosbeta*Pl_12[l] ) 
        F2l2 = Nl2[l] * 2 * ( -(l-1)*cosbeta*Pl2[l] + (l+2)*Pl_12[l] )  
        F1l0 = Nl0[l] * ( -((l-0**2) + 0.5*l*(l-1)*(sinbeta**2))*Pl0[l] + (l+0)*cosbeta*Pl_10[l] )/sinbeta**2 
        F1l0[where0] = Nl0[l] * ( -(0.5*l*(l-1))*Pl0[l][where0] )
        print (Pl0[l][where0])

        # new formulae for diagonal (hybrid)
        #F1l2_D = Nl2[l] * ( -((l-2**2) + 0.5*l*(l-1)*(sinbeta_D**2))*Pl2_D[l] + (l+2)*cosbeta_D*Pl_12_D[l] ) 
        #F2l2_D = Nl2[l] * 2 * ( -(l-1)*cosbeta_D*Pl2_D[l] + (l+2)*Pl_12_D[l] )  

        #F1l2[where0] = F1l2_D
        #F2l2[where0] = F2l2_D

        F2l0 = 0 

        Wl11.append(norm_l * (F1l2 * cos2psi1 * cos2psi2 - F2l2 * sin2psi1 * sin2psi2))
        Wl22.append(norm_l * (F1l2 * sin2psi1 * sin2psi2 - F2l2 * cos2psi1 * cos2psi2))
        Wl12.append(norm_l * 1j * (F1l2 * sin2psi1 * cos2psi2 + F2l2 * cos2psi1 * sin2psi2))
        Wl21.append(norm_l * -1j * (F1l2 * cos2psi1 * sin2psi2 + F2l2 * sin2psi1 * cos2psi2))
        Wl01.append(norm_l * F1l0 * cos2psi2)
        Wl02.append(norm_l * -1j * F1l0 * sin2psi2)

    logger.debug(f'time for W factors calculations: {time.time()-t0}')

    Wl11 = np.array(Wl11)
    Wl22 = np.array(Wl22)
    Wl12 = np.array(Wl12)
    Wl21 = np.array(Wl21)
    Wl01 = np.array(Wl01)
    Wl02 = np.array(Wl02)

    if isinstance(prename, str):
        try:
            if not os.path.isdir('./precomputed'):
                os.mkdir('./precomputed')
            np.savez_compressed(prename, Wl11=Wl11, Wl22=Wl22, Wl01=Wl01, Wl02=Wl02, Wl12=Wl12, Wl21=Wl21)
        except:
            print ('invalid prename is given.')

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

    if len(np.shape(cls)) == 2:
        cls = cls[0]

    cls = cls[:lmax+1]
    if (type(pls)==type(None)):
        pls = gen_Pl_ana(nside, lmax)

    ## l sum with (2l+1)Cl/4pi -> covariance
    ell = np.arange(len(cls))
    bls = (2.*ell+1)/4/np.pi * cls
    cov = np.einsum('l,lij->ij', bls[:lmax+1], pls[:lmax+1])

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
            print (f'The file {prename} does not exist. Generating...')
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
            print (f'The file {prename} does not exist. Generating...')
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
    pls = pls[:lmax+1]
    Wl11 = Wl11[:lmax+1]
    Wl12 = Wl12[:lmax+1]
    Wl21 = Wl21[:lmax+1]
    Wl22 = Wl22[:lmax+1]
    Wl01 = Wl01[:lmax+1]
    Wl02 = Wl02[:lmax+1]

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

    #cov = [[covTT, covTQ, covTU], [covTQ.T, covQQ, cov00], [covTU.T, cov00, covUU]]
    #cov = [[covTT, covTQ, covTU], [covTQ.T, covQQ, covQU.T], [covTU.T, covUQ.T, covUU]]
    cov = [[covTT, covTQ, covTU], [covTQ.T, covQQ, covUQ], [covTU.T, covUQ.T, covUU]]
    cov = np.concatenate(np.concatenate(cov, 1),1)

    return cov.real
 

