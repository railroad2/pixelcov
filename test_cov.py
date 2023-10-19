import os 
import sys
import unittest

import numpy as np
import healpy as hp
import pylab as plt

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from pixelcov import covmat_ana, covmat_est, covmat_nk, spectrum
from pixelcov.utils import dl2cl
from pixelcov.vis_covmat import show_cov


class covariance_calculation_test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nside = 2
        cls.npix = 12*cls.nside*cls.nside
        cls.lmax = 3*cls.nside-1
        cls.dl = np.zeros(cls.lmax+1)+1
        cls.dl[0] = 0
        cls.dl[1] = 0
        cls.dl = np.array([cls.dl, cls.dl*1, cls.dl*0, cls.dl*1])
        print ('Dls:\n',cls.dl)
        cls.cl = dl2cl(cls.dl)
        cls.ell = np.arange(len(cls.dl[0]))
        cls.diag_expect = np.sum((2.*cls.ell+1)/4/np.pi*cls.cl[0])
        print ('Expected diagonal component( sum_l (2l+1)Cl/(4pi) ):', cls.diag_expect)
        cls.cov_ana = []
        cls.cov_est = []

    """
    def test1_ana(self):
        nside = self.nside
        lmax = self.lmax
        npix = self.npix
        cov = covmat_ana.getcov_ana_pol(self.dl, nside=nside, lmax=lmax, isDl=True)
        diag_actual = np.average(np.diagonal(cov[:npix,:npix]))
        print ('covariance matrix for constant Dl with Nside=',nside)
        print ('Shape of covariance matrix:', cov.shape)
        print ('average of diagonal terms of TT block:', diag_actual)
        show_cov(cov, title='Analytic', logscale=False)
        self.cov_ana = cov


    def test2_est(self):
        nside = self.nside
        lmax = self.lmax 
        npix = self.npix
        cov = covmat_est.getcov_est(self.dl, nside, lmax=lmax, nsample=10000, isDl=True, pol=True)
        diag_actual = np.average(np.diagonal(cov[:npix,:npix]))
        print ('covariance matrix for constant Dl with Nside=',nside)
        print ('Shape of covariance matrix:', cov.shape)
        print ('average of diagonal terms of TT block:', diag_actual)
        show_cov(cov, title='Estimation', logscale=False)
        self.cov_est = cov
        

    def test3_nk(self):
        nside = self.nside
        lmax = self.lmax 
        cov = covmat_nk.getcov_nk(self.dl[0], nside=nside, lmax=lmax, isDl=True)
        diag_actual = np.average(np.diagonal(cov))
        print ('covariance matrix for constant Dl with Nside=',nside)
        print ('Shape of covariance matrix:', cov.shape)
        print ('average of diagonal terms of TT block:', diag_actual)
        show_cov(cov, title='NK', logscale=True)

    """

    def test4_compare(self):

        nside = self.nside
        lmax = self.lmax
        npix = self.npix
        cov = covmat_ana.getcov_ana_pol(self.dl, nside=nside, lmax=lmax, isDl=True)
        diag_actual = np.average(np.diagonal(cov[:npix, :npix]))
        print ('covariance matrix for constant Dl with Nside=',nside)
        print ('Shape of covariance matrix:', cov.shape)
        print ('average of diagonal terms of TT block:', diag_actual)
        show_cov(cov, title='Analytic', logscale=False)
        cov_ana = cov
        diag_ana = diag_actual

        nside = self.nside
        lmax = self.lmax 
        npix = self.npix
        cov = covmat_est.getcov_est(self.dl, nside, lmax=lmax, nsample=100000, isDl=True, pol=True)
        diag_actual = np.average(np.diagonal(cov[:npix, :npix]))
        print ('covariance matrix for constant Dl with Nside=',nside)
        print ('Shape of covariance matrix:', cov.shape)
        print ('average of diagonal terms of TT block:', diag_actual)
        show_cov(cov, title='Estimation', logscale=False)
        cov_est = cov
        diag_est = diag_actual

        cov_diff = cov_est - cov_ana / diag_ana * diag_est
        print(cov_diff.shape)
        show_cov(cov_diff, title='ana - est', logscale=False)

        cov_diff_est = cov_diff/cov_est 
        cov_diff_ana = cov_diff/cov_ana

        cov_diff_est[np.abs(cov_diff_est)>0.5] = 0
        cov_diff_ana[np.abs(cov_diff_ana)>20] = 0

        show_cov(cov_diff_est, title='(ana - est)/est', logscale=False)
        show_cov(cov_diff_ana, title='(ana - est)/ana', logscale=False)

    
    @classmethod
    def tearDownClass(cls):
        plt.show()


if __name__=='__main__':
    unittest.main()



