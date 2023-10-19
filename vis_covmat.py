import numpy as np
import pylab as plt

def show_cls(cls, ofname=None, Dl=False, newfig=True):
    ell = np.arange(len(cls[0]))

    if (newfig):
        plt.figure()

    #plt.rc('text', usetex=True)
    plt.rc('font', family='sans-serif')

    if (Dl):
        plt.loglog(ell, cls[0])
        plt.loglog(ell, cls[1])
        plt.loglog(ell, cls[2])
    else:
        plt.loglog(ell, (ell*(ell+1)/2./np.pi)*cls[0])
        plt.loglog(ell, (ell*(ell+1)/2./np.pi)*cls[1])
        plt.loglog(ell, (ell*(ell+1)/2./np.pi)*cls[2])

    plt.xlabel(r'Multipole moment, $l$')
    plt.ylabel(r'$\frac{l(l+1)}{2\pi} C_l$', fontsize=12)
    plt.legend(['TT', 'EE', 'BB'])

    if (not ofname==None):
        plt.savefig(fname)


def show_cov(cov, ofname=None, title=None, logscale=False):
    if logscale: 
        plt.matshow(np.log10(np.abs(cov)))
    else:
        plt.matshow(cov)

    if (title != None):
        plt.title(title)

    plt.colorbar()


def show_cov_part(cov, part=None, title=None, logscale=False):
    npix = int(len(cov)/3)

    if not title is None:
        title = ''

    TQU = ['T', 'Q', 'U']
    TQUDIC = {'T':0, 'Q':1, 'U':2}

    if part is None:
        for i in range(3):
            for j in range(3):
                covpart = np.asarray(cov)[i*npix:(i+1)*npix-1, j*npix:(j+1)*npix-1]
                show_cov(covpart, title=title+TQU[i]+TQU[j], logscale=logscale)
    else:
        if len(part) != 2:
            print_error('Invalid part string')  
            return

        i = TQUDIC[part[0]]
        j = TQUDIC[part[1]]

        covpart = np.asarray(cov)[i*npix:(i+1)*npix-1, j*npix:(j+1)*npix-1]
        show_cov(covpart, title=title+TQU[i]+TQU[j], logscale=logscale)


