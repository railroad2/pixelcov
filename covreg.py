import numpy as np


def covreg_I(cov, lamb=1e-10): ## regularize with an identity matrix
    l = len(cov)
    II = np.eye(l)
    covr = cov + lamb*II
    
    return covr


def covreg_1(cov, lamb=1e-10): ## regularize with a constant matrix
    l = len(cov)
    CC = np.full((l,l), 1.0)
    covr = cov + lamb*CC
    
    return covr


def covreg_D(cov, lamb=1e-10): ## regularize with a random diagonal matrix
    l = len(cov)
    DD = np.diag(np.random.random(l))
    covr = cov + lamb*DD
    
    return covr


def covreg_R(cov, lamb=1e-10): ## regularize with a random matrix
    l = len(cov)
    RR = np.random.random((l,l))
    covr = cov + lamb*RR
    
    return covr


def covreg_none(cov, lamb=1e-10):
    return cov


def regmat_I(dim, lamb=1e-10): ## regularize with an identity matrix
    l = dim 
    II = np.eye(l)
    covr = lamb*II
    
    return covr


def regmat_1(dim, lamb=1e-10): ## regularize with a constant matrix
    l = dim 
    CC = np.full((l,l), 1.0)
    covr = lamb*CC
    
    return covr


def regmat_D(dim, lamb=1e-10): ## regularize with a random diagonal matrix
    l = dim 
    DD = np.diag(np.random.random(l))
    covr = lamb*DD
    
    return covr


def regmat_R(dim, lamb=1e-10): ## regularize with a random matrix
    l = dim 
    RR = np.random.random((l,l))
    covr = lamb*RR
    
    return covr


def getregmat(dim, lamb, regtype=None):
    if regtype == 'I':
        fnc = regmat_I
    elif regtype == '1':
        fnc = regmat_1
    elif regtype == 'D':
        fnc = regmat_D
    elif regtype == 'R':
        fnc = regmat_R
    else:
        return 0

    regmat = fnc(dim, lamb)

    return regmat


def regularize(cov, lamb, regtype=None):
    dim = len(cov)
    regmat = getregmat(dim, lamb, regtype)

    return cov + regmat


