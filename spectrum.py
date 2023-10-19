import numpy as np
import healpy as hp
import camb

from .utils import *


args_cosmology = ['H0', 'cosmomc_theta', 'ombh2', 'omch2', 'omk', 
                  'neutrino_hierarchy', 'num_massive_nutrinos',
                  'mnu', 'nnu', 'YHe', 'meffsterile', 'standard_neutrino_neff', 
                  'TCMB', 'tau', 'deltazrei', 'bbnpredictor', 'theta_H0_range'] 

args_InitPower = ['As', 'ns', 'nrun', 'nrunrun', 'r', 'nt', 'ntrun', 'pivot_scalar', 
                  'pivot_tensor', 'parameterization']

def get_spectrum_camb(lmax, isDl=True, norm='uK2', cambres=False, **kwargs):
    """
    """
    ## arguments to dictionaries
    kwargs_cosmology={}
    kwargs_InitPower={}
    kwargs_cosmology['H0'] = 67.5
    for key, value in kwargs.items():  # for Python 3, items() instead of iteritems()
        if key in args_cosmology: 
            kwargs_cosmology[key]=value
        elif key in args_InitPower:
            kwargs_InitPower[key]=value
        else:
            print_warning('Wrong keyword: ' + key)

    ## call camb
    pars = camb.CAMBparams()
    pars.set_cosmology(**kwargs_cosmology)
    pars.InitPower.set_params(**kwargs_InitPower)
    pars.WantTensors=True
    results = camb.get_results(pars)
    dls = results.get_total_cls(lmax=lmax).T
    if (norm=='uK2'):
        dls = dls * pars.TCMB**2 * 1e12
    elif (norm=='Tcmb'):
        dls = dls * pars.TCMB**2
    else:
        if (type(norm)!=str):
            dls = dls * norm
        else:
            print_warning('get_spectrum_camb(): Undefined normalization type:'+ str(norm))

    if (isDl):
        res = dls
    else:
        res = dl2cl(dls) 

    if (cambres):
        return res, results
    else:
        return res
    
def get_spectrum_const(lmax, isDl=True):
    """
    """
    dls = np.zeros(lmax+1)+1      
    dls[0] = 0
    dls[1] = 0 

    if (isDl):
        return dls   
    else:
        ell = np.arange(len(dls))
        cls = dls.copy()
        cls[1:] = cls[1:] * 2 * np.pi / ell[1:] / (ell[1:]+1)
        return cls

def get_spectrum_map(mapT, lmax=2000, isDL=False): 
    """
    """
    cls = hp.anafast(mapT, lmax=lmax)
    
    if (isDL):
        ell = np.arange(len(cls)) 
        dls = cls * ell * (ell+1) / 2 / np.pi
        return dls
    else:
        return cls

