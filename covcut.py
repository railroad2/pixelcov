import numpy as np


def cutmap(m, mask):
    if mask == []:
        return m

    if len(m) == 3:
        m_cut = []
        for mm in m:
            m_cut.append(mm[mask==1])
        m_cut = np.array(m_cut)
    else:
        mdim = len(m)
        ndim = len(mask)
        nit = mdim // ndim
        mfull = np.array(list(mask)*nit) 
        m_cut = m[mfull==1]

    return m_cut


def cutcov(cov, mask=[]): 
    if mask == []:
        return cov

    cov = np.array(cov)
    mdim = len(cov)
    ndim = len(mask)
    nit = mdim // ndim
    mfull = np.array(list(mask)*nit) 
    cov_cut = cov[mfull==1][:,mfull==1]
     
    return cov_cut
     

def cutpls(pls, mask):
    pls = np.array(pls)
    mdim = len(pls[0])
    ndim = len(mask)
    mfull = np.array(mask)
    pls_tmp = pls[:,mfull==1][:,:,mfull==1]
     
    return pls_tmp


def cutwls(wls, mask):
    wls = np.array(wls)
    wls_tmp = []
    mdim = len(wls[0][0])
    ndim = len(mask)
    mfull = np.array(mask)
    for i in range(len(wls)):
        wls_tmp.append(wls[i][:,mfull==1][:,:,mfull==1])
     
    return np.array(wls_tmp)


def partmap(m, maptype):
    if maptype is None:
        return m 

    dim = len(m)//3
    # choose covariance blocks according to the maptype
    if   (maptype == 'TQU'):
        res = m 
    elif (maptype == 'T'):
        res = m[:dim]
    elif (maptype == 'Q'):
        res = m[dim:dim*2]
    elif (maptype == 'U'):
        res = m[dim*2:dim*3]
    elif (maptype == 'TQ'):
        res = m[:dim*2]
    elif (maptype == 'TU'):
        res = np.delete(m, np.s_[dim:dim*2])
    elif (maptype == 'QU'):
        res = m[dim:dim*3]

    return res


def partcov(cov, maptype=None, covblk=[]):
    if (maptype is None) and (covblk == []):
        return cov
    else:
        if maptype is None:
            maptype = 'TQU'

        if covblk == []:
            covblk = ['TT', 'QQ', 'UU', 'TQ', 'TU', 'QU']

    dim = len(cov)//3 

    m_0 = np.zeros((dim, dim))
    covp = cov.copy()

    if (len(covblk)==0):
        print_error('define the covariance blocks to be used')
        return

    # choose covariance blocks
    if not('TT' in covblk):
        covp[:dim, :dim] = m_0
    if not('QQ' in covblk):
        covp[dim:dim*2, dim:dim*2] = m_0
    if not('UU' in covblk):
        covp[dim*2:dim*3, dim*2:dim*3] = m_0
    if not('TQ' in covblk):
        covp[:dim, dim:dim*2] = m_0
        covp[dim:dim*2, :dim] = m_0
    if not('TU' in covblk):
        covp[:dim, dim*2:dim*3] = m_0
        covp[dim*2:dim*3, :dim] = m_0
    if not('QU' in covblk):
        covp[dim*2:dim*3, dim:dim*2] = m_0
        covp[dim:dim*2, dim*2:dim*3] = m_0

    # choose covariance blocks according to the maptype
    if   (maptype == 'TQU'):
        res = covp
    elif (maptype == 'T'):
        res = covp[:dim, :dim]
    elif (maptype == 'Q'):
        res = covp[dim:dim*2, dim:dim*2]
    elif (maptype == 'U'):
        res = covp[dim*2:dim*3, dim*2:dim*3]
    elif (maptype == 'TQ'):
        res = covp[:dim*2, :dim*2]
    elif (maptype == 'TU'):
        res = np.delete(np.delete(covp, np.s_[dim:dim*2], 0), np.s_[dim:dim*2], 1)
    elif (maptype == 'QU'):
        res = covp[dim:dim*3, dim:dim*3]

    return res


def Kmat(nside, lmin=None, fsky=None):
    import healpy as hp
    ## a matrix to filter the larger angular scale given fsky 
    if lmin is None:
        if not (fsky is None):
            lmin = int(1./fsky - 0.5)
        else:
            print ('One of lmin or fsky must be given.')

    if lmin == 0: lmin = 1

    angmax = np.pi/lmin

    npix = hp.nside2npix(nside)
    ipix = np.arange(12*nside**2)
    vec = np.array(hp.pix2vec(nside, ipix)).T
    vgrid1 = np.array([vec]*len(vec))
    vgrid2 = np.transpose(vgrid1, (1,0,2))
    dot = np.sum(vgrid1*vgrid2, axis=-1)
    dot[dot>1]=1; dot[dot < -1]=-1
    ang = np.arccos(dot)
    K = np.ones(ang.shape)
    K[ang>angmax]=0

    return K

