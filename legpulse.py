import numpy as np
import scipy.special as sp
import healpy as hp
import matplotlib.pyplot as plt

def legpulse(npeak, nlook, Nmax):
    lpulse = 0
    npeak /= np.linalg.norm(npeak)
    nlook /= np.linalg.norm(nlook)
    ctheta = np.dot(npeak, nlook)
    for n in range(Nmax+1):
        pn = sp.eval_legendre(n, ctheta)
        lpulse += pn * (2 * n + 1) / (4 * np.pi)
    return lpulse

def legvec(npeak, narr, Nmax):
    lvec = np.zeros((np.shape(narr)[0], ))
    for ind in range(np.shape(narr)[0]):
        lvec[ind] = legpulse(npeak, narr[ind, :], Nmax)
    return lvec/np.linalg.norm(lvec)

def legdict(orderpeak, ordergrid, Nmax):
    peakvecs, _, peakpix = grid(orderpeak)
    gridvecs, _, gridpix = grid(ordergrid)
    legdict = np.zeros((gridpix, peakpix))
    for indpeak in range(peakpix):
        legdict[:, indpeak] = legvec(peakvecs[indpeak,:], gridvecs, Nmax)
    return legdict

def legdict_common(lev, comvecs, npixcom, Nmax, micpos):
    #grid(orderpeak)
    gridvecs, _, gridpix = grid(lev)
    legdict = np.zeros((npixcom,gridpix))
    for indpeak in range(npixcom):
        legdict[indpeak,:] = legvec(comvecs[indpeak,:]-micpos, gridvecs, Nmax)
    return legdict


def grid(order):
    npix = hp.order2npix(order)
    nside = hp.order2nside(order)
    gvec = np.zeros((npix, 3))
    alist = []
    for ipix in range(npix):
        gvec[ipix, :] = hp.pix2vec(nside, ipix, nest=True)
        alist.append(hp.pix2ang(nside,ipix, nest=True))
    return gvec, alist, npix

if __name__ == '__main__':
    ld = legdict(3, 3, 3)
    pix = hp.ang2pix(hp.order2nside(3), np.pi/2, 0, nest=True)
    hp.mollview(ld[:,pix], nest=True)
    plt.show()
    pass
