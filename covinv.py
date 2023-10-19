import numpy as np
import sys
from numpy.linalg   import inv, pinv, slogdet, svd


def detinv_pseudo(cov):
    """

    """
    u, s, vh = svd(cov)

    uh = np.matrix(u).H
    v  = np.matrix(vh).H
    si = pinv(np.diagflat(s))
    tol = sys.float_info.epsilon

    covi = np.matmul(v,np.matmul(si, uh))
    plogdet = np.sum(np.log(s[s > tol]))

    return plogdet, covi


def detinv(cov):
    s, logdet = slogdet(cov)
    covi = inv(cov)

    return logdet, covi


