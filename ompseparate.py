import numpy as np
from scipy.special import legendre, eval_legendre#, sph_jnyn
import healpy as hp
import matplotlib.pyplot as plt
import time as tm
#import scenegen as sgen
import pickle as pkl
from compomp import *
import healpy as hp
import scipy.special as spec
#import Microphone as mic
import os

# Utility functions
def sph2cart(r, th, ph):
    x = r * np.sin(th) * np.cos(ph)
    y = r * np.sin(th) * np.sin(ph)
    z = r * np.cos(th)
    u = np.array([x, y, z])
    return u

def cart2sph(x, y, z):
    """
    r, th, ph = cart2sph(x, y, z)

    Return the spherical coordinate representation of point(s) given in
    Cartesian coordinates

    As usual r is the radius, th is the elevation angle defined from the
    positive z axis and ph is the azimuth angle defined from the positive x axis
    """
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    th = np.arccos(z / r)
    ph = np.arctan2(y, x)
    return r, th, ph

def getYnmtlookup(filepath):
    f = open(filepath, 'rb')
    Ynmt = pkl.load(f)
    f.close()
    return Ynmt


# Calculate dictionary atoms
def legendrepulse(Ndec, lev, thetas, phis):
    '''Returns the value of the Legendre pulse at all pixels for an (Nside) HP grid for a given source direction'''
    Nside = 2 ** lev
    numpix = hp.nside2npix(Nside)
    nsrc = sph2cart(1., thetas, phis)
    cosTheta = []
    for indx in range(numpix):
        th, ph = hp.pix2ang(Nside, indx, nest=True)
        npix = sph2cart(1., th, ph)
        cosTheta.append(np.dot(nsrc, npix))
    costh = np.array(cosTheta)
    pls = np.zeros((numpix,))
    for n in range(Ndec + 1):
        pls += (2 * n + 1) / (4 * np.pi) * eval_legendre(n, costh)
    nrm = pls/np.linalg.norm(pls)
    # pls /= (Ndec + 1) * 4 * np.pi
    return pls, nrm


# Calculate dictionary to be used in OMP
def legendredict(Ndec, levdict, levgran):
    assert (levgran >= levdict)
    ngran = 2 ** levgran
    dpix = hp.nside2npix(ngran)
    ndict = 2 ** levdict
    ddict = hp.nside2npix(ndict)
    D = np.zeros((ddict, dpix), dtype=complex)
    for indp in range(dpix):
        th, ph = hp.pix2ang(ngran, indp, nest=True)
        D[:, indp], _ = legendrepulse(Ndec, levdict, th, ph)
    return D

def legendredict2(Ndec, levdict, centers):
    # calibration source and mics are in horizontal plane
    #centers is list of tuples (theta,phi)

    ndict = 2 ** levdict
    ddict = hp.nside2npix(ndict)
    D = np.zeros((ddict, len(centers)), dtype=complex)
    ind = 0
    for tp in centers:
        th, ph = tp[0] , tp[1]
        D[:, ind], _ = legendrepulse(Ndec, levdict, th, ph)
        ind += 1
    return D
def azimuth_directions(num,sector=(0,2*np.pi),flag='uniform'):
    # calibration source and mics are in horizontal plane
    outlist = []
    if flag == 'uniform':
        phis = np.linspace(sector[0],sector[1], num+1)[:-2]
    for phi in phis:
        outlist.append((np.pi/2,phi))
    return outlist

def get_azimuthvector(ind, num, sector, flag='uniform'):
    # calibration source and mics are in horizontal plane
    if flag == 'uniform':
        phis = np.linspace(sector[0],sector[1], num+1)[:-2]
        phi = phis(ind)
        vec = np.array([np.cos(phi), np.sin(phi), 0])
    return vec

# WORK ON THIS!!!!!!!!!!
def legendredict_sc(Ndec, levdict, levgran, doas):
    assert (levgran >= levdict)
    depth = levgran - levdict
    dicterms = set()
    mult = 4**depth
    increment = np.array(range(4**depth))

    for item in doas:
        ths, phs = item
        slf = hp.ang2pix(2**levdict, ths, phs, nest=True)
        dicterms = dicterms.union(slf * mult + increment)
        ngh = hp.get_all_neighbours(2**levdict, ths, phs, nest=True)
        for item in ngh:
            if item >= 0:
                dicterms = dicterms.union(item * mult + increment)

    dt = list(dicterms)
    dpix = len(dicterms)
    ndict = 2 ** levdict
    ddict = hp.nside2npix(ndict)
    nrm = []
    D = np.zeros((ddict, dpix), dtype=complex)
    for indp in range(dpix):
        th, ph = hp.pix2ang(2**levgran, dt[indp], nest=True)
        D[:, indp], nrmo = legendrepulse(Ndec, levdict, th, ph)
        nrm.append(nrmo)
    return D, dt, np.array(nrm)

# Steered response functional at a single point
def srfunc(mvec, k, ra, Nmax, lvl, idx, ydict):
    assert np.size(mvec) == (Nmax + 1) ** 2
    kra = k * ra
    jn, jnp, yn, ynp = sph_jnyn(Nmax, kra)
    hn = jn - 1j * yn
    hnp = jnp - 1j * ynp
    bnkra = jn - (jnp / hnp) * hn
    b = []
    for n in range(Nmax + 1):
        for count in range(-n, n + 1):
            b.append(1 / (4 * np.pi * (1j) ** n * bnkra[n]))
    b = np.array(b)
    # ns = np.array(ns)
    mvec = mvec * b
    yd = ydict[(lvl, idx)][:(Nmax + 1) ** 2]
    srfval = np.sum(mvec * yd)
    return srfval

# Steered response functional at all points at a given resolution
def srf(mvec, k, ra, Nmax, lvl, ydict):
    imax = hp.nside2npix(2 ** lvl)
    srfvect = np.zeros(imax, dtype=complex)
    for ix in range(imax):
        srfvect[ix] = srfunc(mvec, k, ra, Nmax, lvl, ix, ydict)
    return srfvect


# Calculate SRF with order selection
def srford(mvec, f, ra, lvl, ydict, Nmax):
    k = 2 * np.pi * f / 340.
    mv = mvec[:(Nmax + 1) ** 2]
    srfvect = srf(mv, k, ra, Nmax, lvl, ydict)
    return srfvect, Nmax


# Select the SHD vector to calculate SRF
def selectbinindx(fidx, tidx, Pnm, Ndec):
    mvec = np.zeros((Ndec + 1) ** 2, dtype=complex)
    for idm in range((Ndec + 1) ** 2):
        M = Pnm[idm]
        mvec[idm] = M[fidx, tidx]
    return mvec

def selectalphas(inds, alphan, suppressgain, radius, doai, interind, gran): # WE ARE NOT CHECKING FOR SOURCES!!! CHECK!!!
    alphas = np.copy(alphan)
    for indint in interind:
        alphas[indint] *= suppressgain
    for idx in range(len(doai)):
        item = doai[idx]
        th = item[0]
        ph = item[1]
        vec = sph2cart(1., th, ph)
        infs = hp.query_disc(2**gran, vec, radius, nest=True)
        for allind in inds:
            if allind in infs:
                # print('alphas modified')
                alphas[inds.index(allind)+len(doai)+1] *= suppressgain ## CHECK HERE IF PROBLEM
    return inds, alphas

def suppressalphas(inds, alphan, suppressgain, radius, doas, indsrc, gran): # WE ARE NOT CHECKING FOR SOURCES!!! CHECK!!!
    alphas = np.copy(alphan)
    for indint in range(len(doas)):
        if indint != indsrc:
            alphas[indint] *= suppressgain
    item = doas[indsrc]
    th = item[0]
    ph = item[1]
    vec = sph2cart(1., th, ph)
    infs = hp.query_disc(2**gran, vec, radius, nest=True)
    for allind in inds:
        if allind not in infs:
            # print('alphas modified')
            alphas[inds.index(allind)+len(doas)] *= suppressgain ## CHECK HERE IF PROBLEM
    return inds, alphas

# def legendreval(Ndec, thetas, phis, thetal, phil):
#     '''Returns the value of the Legendre pulse at for a given steering direction'''
#     nsrc = sph2cart(1., thetas, phis)
#     nsteer = sph2cart(1., thetal, phil)
#     costh = np.dot(nsrc, nsteer)
#     val = 0
#     for n in range(Ndec + 1):
#         val += (2 * n + 1) / (4 * np.pi) * eval_legendre(n, costh)
#     return val

def beamformexact(alphas, inds, thl, phl, Nmax, gran):
    out = 0j
    for indi in range(len(inds)):
        th, ph = hp.pix2ang(2**gran, inds[indi], nest=True)
        vl = legendreval(Nmax, th, ph, thl, phl)
        out += vl * alphas[indi]
    return out


# def beamform(Psi, alphas, thl, phl, lev): #, Ndec, gran, doas):
#     bf = Psi * alphas
#     idxbf = hp.ang2pix(2**lev, thl, phl, nest=True)
#     out = bf[idxbf]
#     # L = len(alphas)
#     # out = 0j
#     # nsrc = len(doas)
#     # for lind in range(nsrc):
#     #     alphal = alphas[lind]
#     #     ths = doas[lind][0]
#     #     phs = doas[lind][1]
#     #     dictval = legendreval(Ndec, ths, phs, thl, phl)
#     #     out += alphal * dictval
#     # for lind in range(nsrc, L):
#     #     alphal = alphas[lind]
#     #     indd = inds[lind-nsrc]
#     #     ths, phs = hp.pix2ang(2**gran, indd, nest=True)
#     #     dictval = legendreval(Ndec, ths, phs, thl, phl)
#     #     out += alphal * dictval
#     return out

def windowcreate(mu=(np.pi/2, 0.), sigma=np.pi/8, lev=1):
    # assert (sigma < np.pi / 2)
    nside = 2 ** lev
    npix = hp.nside2npix(nside)
    nmu = sph2cart(1., mu[0], mu[1])
    gauss = []
    for indpix in range(npix):
        th, ph = hp.pix2ang(nside, indpix, nest=True)
        npix = sph2cart(1., th, ph)
        thdiff = np.arccos(np.dot(npix, nmu))
        # gauss.append(np.exp(-0.5 * (thdiff / sigma) ** 2))
        gauss.append(np.exp(-0.5 * (thdiff / sigma) ** 2))
    return gauss

def windowmask(targetdoa, interferencedoas, sigma, lev):
    wtarget = windowcreate(targetdoa, sigma, lev)
    wtarget = np.array(wtarget)
    for item in interferencedoas:
        print(item)
        wint = windowcreate(item, sigma, lev)
        wtarget *= (1-np.array(wint))
    wtarget /= np.max(wtarget)
    return list(wtarget)

def bwwin(doa, thc, N, lev, suppressdB):
    nside = 2 ** lev
    npix = hp.nside2npix(nside)
    nmu = sph2cart(1., doa[0], doa[1])
    bwin = []
    for indpix in range(npix):
        th, ph = hp.pix2ang(nside, indpix, nest=True)
        npix = sph2cart(1., th, ph)
        thdiff = np.arccos(np.dot(npix, nmu))
        bwin.append(1/np.sqrt(1+(thdiff/thc)**(2*N)))
    k = 10**(suppressdB/20.)
    print(k)
    wn = np.array(bwin) * (1-k) + k
    return wn

def windowg(svec, Psinv, win):
    alphas = Psinv * np.matrix(win * svec).T
    return alphas

# Windowing with higher granularity at the activation level
def windowalphas(doas, alphas, inds, win, gran):
    inddoas = range(len(doas))
    for indd in inddoas:
        th = doas[indd][0]
        ph = doas[indd][1]
        indx = hp.ang2pix(2**gran, th, ph, nest=True)
        alphas[indd] *= win[indx]
    lendoa = len(doas)
    for ind in range(lendoa, len(alphas)):
        alphas[ind] *= win[inds[ind-lendoa]]
    return alphas

def alphamask(alphas, inds, levdict, levgran, doa):
    indself = hp.ang2pix(2**levdict, doa[0], doa[1], nest=True) # The pixel at which the DOA resides
    indneig = hp.get_all_neighbours(2**levdict, doa[0], doa[1], nest=True) # Neighbours of that pixel
    b = set()
    b = b.union([indself])
    b = b.union(indneig)
    indchk = list(b)

    depthcoef = 4**(levgran-levdict)
    levdct = np.array(inds) // depthcoef # Find the pixels for the identified dictionary elements at levdict

    mask = np.zeros((len(inds),), dtype=bool)
    for ind in indchk:
        mask |= np.array(levdct == ind)
    # print(mask)
    for ind in range(len(mask)):
        alphas[ind] *= mask[ind]
    return alphas

def planewave(freq, ra, A, thetas, phis, Nmax):
    mvec = []
    kra = 2 * np.pi * freq / 340. * ra
    jn, jnp, yn, ynp = spec.sph_jnyn(Nmax, kra)
    hn = jn - 1j * yn
    hnp = jnp - 1j * ynp
    bnkra = jn - (jnp / hnp) * hn
    b = []
    for n in range(Nmax+1):
        b.append(bnkra[n])

    for n in range(Nmax+1):
        for m in range(-n, n+1):
            pnm = A * np.conj(spec.sph_harm(m, n, phis, thetas))
            mvec.append(pnm)
    return np.array(mvec)

# def getYs(lev, Ndec):
#     npix = hp.nside2npix(2 ** lev)
#     Ys = np.zeros((npix, (Ndec+1)**2), dtype=complex)
#
#     for ind in range(npix):
#         th, ph = hp.pix2ang(2**lev, ind, nest=True)
#         jnd = 0
#         for n in range(5):
#             for m in range(-n,n+1):
#                 Ys[ind, jnd] = spec.sph_harm(m,n,ph,th)
#                 jnd += 1
#     return np.matrix(Ys)

def getP(doapix, Ndec, lev):
    D = legendredict(Ndec, lev, lev)
    npix = hp.nside2npix(2**lev)
    P = np.zeros((npix, len(doapix)))
    ind = 0
    for item in doapix:
        P[:,ind] = np.real(D[:,item])
        ind += 1
    return np.matrix(P)

def getmvec(doapix, lev, freq, ra, alphas, Ndec):
    mv = np.zeros(((Ndec+1)**2,), dtype=complex)
    ind = 0
    for pix in doapix:
        th, ph = hp.pix2ang(2**lev, pix, nest=True)
        mv += planewave(freq, ra, alphas[ind], th, ph, Ndec)
        ind += 1
    return mv

def pwnoise(num, freq, ra, dB, Ndec):
    lin = 10**(dB/10)
    nsn = np.random.rand(num) + 1j * np.random.rand(num)
    alphan = nsn * lin
    mvn = np.zeros(((Ndec + 1) ** 2,), dtype=complex)
    for ind in range(num):
        th = np.random.rand(1) * np.pi
        ph = np.random.rand(1) * 2 * np.pi
        ns = planewave(freq, ra, alphan[ind], th, ph, Ndec)
        ns = np.reshape(ns, (25,))
        mvn += ns
    return mvn

def calculatetemplatealphas(doapix, numnoise, dB, Ndec, ra, lev, alphas, freq):
    P = getP(doapix, Ndec, lev)
    Ys = getYs(lev, Ndec)
    mv = getmvec(doapix, lev, freq, ra, alphas, Ndec)
    mvn = pwnoise(numnoise, freq, ra, dB, Ndec)
    mv += mvn
    yN = np.matrix(Ys) * np.matrix(mv).T
    alphabar = np.linalg.pinv(P) * yN * 0.18251713657327448
    return np.array(alphabar)

def calculatesralphas(doapix, numnoise, dB, lev, freq, ra, alphas, Ndec, spix):
    ths, phs = hp.pix2ang(2**lev, spix, nest=True)
    mv = getmvec(doapix, lev, freq, ra, alphas, Ndec)
    mvn = pwnoise(numnoise, freq, ra, dB, Ndec)
    mv += mvn
    y = []
    for n in range(Ndec+1):
        for m in range(-n,n+1):
            y.append(spec.sph_harm(m, n, phs, ths))
    yH = np.sum(np.array(y)*mv)/(4*np.pi)*(Ndec+1)*1.142857142857143
    return yH

def error(doapix, numnoise, dB, lev, freq, ra, alphas, Ndec):
    a = calculatetemplatealphas(doapix, numnoise, dB, Ndec, ra, lev, alphas, freq)
    ars = []
    for ind in range(len(doapix)):
        ay = calculatesralphas(doapix, numnoise, dB, lev, freq, ra, alphas, Ndec, doapix[ind])
        ars.append(ay)
    ars = np.array(ars)
    atp = np.array(a)
    alp = np.array(alphas)
    errsr = np.sum(np.abs(ars - alp)) / len(alp)
    errtp = np.sum(np.abs(atp - alp)) / len(alp)
    return errtp, errsr

def getBi(k, ra, Nmax):
    kra = k * ra
    jn, jnp, yn, ynp = sph_jnyn(Nmax, kra)
    hn = jn - 1j * yn
    hnp = jnp - 1j * ynp
    bnkra = jn - (jnp / hnp) * hn
    b = []
    for n in range(Nmax + 1):
        for count in range(-n, n + 1):
            b.append(1. / (4 * np.pi * (1j) ** n * bnkra[n]))
    Bi = np.matrix(np.diag(b))
    return Bi

def getYQH(thm, phm, Nmax):
    Ynm = []
    for ind in range(len(thm)):
        for n in range(Nmax+1):
            for m in range(-n, n+1):
                Ynm.append(spec.sph_harm(m, n, phm[ind], thm[ind]))
    YQ = np.array(Ynm)
    YQ = np.reshape(YQ, (32, (Nmax+1)**2))
    YQH = np.matrix(np.conj(YQ)).T
    return YQH

def getYS(Nmax, lev):
    Ynm = []
    npix = hp.nside2npix(2**lev)
    for ipix in range(npix):
        th, ph = hp.pix2ang(2**lev, ipix, nest=True)
        for n in range(Nmax+1):
            for m in range(-n, n+1):
                Ynm.append(spec.sph_harm(m, n, ph, th))
    YS = np.array(Ynm)
    YS = np.reshape(YS, (npix, (Nmax + 1) ** 2))
    return np.matrix(YS)

def getW(mic):
    W = np.diag(mic._weights)
    return np.matrix(W)

def getPi(doas, lev, Ndec):
    npix = hp.nside2npix(2 ** lev)
    P = np.zeros((npix, len(doas)))
    for ind in range(len(doas)):
        # px = hp.ang2pix(2**lev, doas[ind][0], doas[ind][1], nest=True)
        # th, ph = hp.pix2ang(2**lev, px, nest=True)
        lp = legendrepulse(Ndec, lev, doas[ind][0], doas[ind][1])
        P[:,ind] = lp
    P = np.matrix(P)
    # Pi = np.linalg.inv(P.T*P)*P.T
    Pi = np.linalg.pinv(P)
    return Pi, P

def getunmixingmatrix(Pi, YS, Bi, YQH):
    S = YS * Bi * YQH
    M = Pi * S # Unmixing matrix
    return M, S

def getsrmatrix(YS, Bi, YQH):
    S = YS * Bi * YQH
    return S

def synthbeamform(doas, targetind, alphas, Ndec):
    alpha = 0j
    for ind in range(len(doas)):
        gain = legendreval(Ndec, doas[targetind][0], doas[targetind][1], doas[ind][0], doas[ind][1])
        alpha += alphas[ind] * gain
    return alpha

def ind2doa(xtuple, ctuple):
    x = np.array(xtuple, dtype=float)
    c = np.array(ctuple, dtype=float)
    v = x - c
    v *= np.array([-.5, -.5, .3])
    _, th, ph = cart2sph(v[0], v[1], v[2])
    return th, ph

def steeringvector(doa, Nmax):
    th = doa[0]
    ph = doa[1]
    Ynm = []
    for n in range(Nmax+1):
        for m in range(-n, n+1):
            Ynm.append(spec.sph_harm(m, n, ph, th))
    ys = np.matrix(Ynm)
    return ys

def beamform(ys, Bi, YQH, s):
    out = np.array(ys * Bi * YQH * s)[0,0]
    return out

if __name__ == '__main__':

    # lev = 2
    # # Nmaxx = 4
    # numsrc = 4
    # samples= 240000
    # fs = 48000
    # NFFT = 1024
    # olap = 512
    #
    # audiofiles = ['speech-01-norm.wav', 'speech-02-norm.wav', 'speech-03-norm.wav', 'speech-04-norm.wav']
    #
    # # Mic_RIRs_path = '/Users/huseyinhacihabiboglu/PycharmProjects/Localisation/em32IRs_3D'
    #
    # # dirFiles = os.listdir(Mic_RIRs_path) #list of directory files
    # # dirFiles.sort() #good initial sort but doesnt sort numerically very well
    # # sorted(dirFiles) #sort numerically in ascending order
    #
    # # print dirFiles
    # # mg = tp3.measgrid()
    # #testinstance = tp3.selectset(mg, numsrc, clustind, numclust, np.pi/4, Ntry)
    # testinstance = ([(3, 1, 2), (1, 3, 2), (5,3,2) ,(3,5,2)], 1.0)
    # cent = (3, 3, 2)
    # doas = []
    # for ind in range(numsrc):
    #     th, ph = ind2doa(testinstance[0][ind], cent)
    #     doas.append((th,ph))
    #
    # print(doas)
    # # doas = [[1.9334830318936342, 0.32175055439664213], [1.9720434687132573, 3.9269908169872414]]
    # # doas = [(np.pi/2.,  0), (np.pi/2.,  np.pi/2.), (np.pi/2.,  np.pi), (np.pi/2.,  3 * np.pi/2.)]
    # print(testinstance[0])
    #
    # sgo = sgen.composescene3d(audiofiles[:numsrc], testinstance[0], samples)
    # A, fi, ti = sgen.preprocessinput(sgo, fs, NFFT, olap)
    #
    # srconly = []
    # for ind in range(numsrc):
    #     print(testinstance[0][ind])
    #     print(audiofiles[ind])
    #     src = sgen.composescene3d([audiofiles[ind]], [testinstance[0][ind]], samples)
    #     SRC, fi, ti = sgen.preprocessinput(src, fs, NFFT, olap)
    #     srconly.append(SRC)
    #
    #
    # Mun = []
    # Sun = []
    # Pun = []
    #
    # em32 = mic.EigenmikeEM32().returnAsStruct()
    # thm = em32['thetas']
    # phm = em32['phis']
    #
    # Nmax = range(5)
    #
    # # for find in range(NFFT/2+1):
    # #     print(find)
    # #     k = 2 * np.pi * fi[find] / 340.
    # #     ra = em32['radius']
    # #     kra = int(np.round(k * ra))
    # #     if kra >= 4:
    # #         kra = 4
    # #     Pi, P = getPi(doas, lev, Nmax[kra])
    # #     YS = getYS(Nmax[kra], lev)
    # #     YQH = getYQH(thm, phm, Nmax[kra])
    # #     Bi = getBi(k, ra, Nmax[kra])
    # #     Mi = getunmixingmatrix(Pi, YS, Bi, YQH)
    # #     Mun.append(Mi)
    #
    # for find in range(NFFT/2+1):
    #     print(find)
    #     k = 2 * np.pi * fi[find] / 340.
    #     ra = em32['radius']
    #     kra = (k * ra)
    #     if kra < 2:
    #         kra = 2
    #     elif kra < 3:
    #         kra = 3
    #     else:
    #         kra =4
    #
    #     Pi, P = getPi(doas, lev, Nmax[kra])
    #     YS = getYS(Nmax[kra], lev)
    #     YQH = getYQH(thm, phm, Nmax[kra])
    #     Bi = getBi(k, ra, Nmax[kra])
    #     Mi, S = getunmixingmatrix(Pi, YS, Bi, YQH)
    #     Mun.append(Mi)
    #     Sun.append(S)
    #     Pun.append(P)
    #     # Mun.append((P, S))
    #     targetind = 0
    # # Pim.append(Pim)
    # # YSm.append(YS)
    # # YQHm.append(YQH)
    #
    # B = []
    # C = []
    # D = []
    # E = []
    # # E = np.zeros(A[0].shape) #This will hold the LS approximation error
    # s = np.matrix(np.zeros((32,1), dtype=complex))
    #
    # for sind in range(len(doas)):
    #     B.append(np.zeros(A[0].shape, dtype=complex))
    #     C.append(np.zeros(A[0].shape, dtype=complex))
    #     D.append(np.zeros(A[0].shape, dtype=complex))
    #     E.append(np.zeros(A[0].shape, dtype=complex))
    #
    # for find in range(1,NFFT/2+1):
    #     print(find)
    #     k = 2 * np.pi * fi[find] / 340.
    #     ra = em32['radius']
    #     kra = (k * ra)
    #     if kra < 2:
    #         kra = 2
    #     elif kra < 3:
    #         kra = 3
    #     else:
    #         kra =4
    #
    #     Mi = Mun[find]
    #     S = Sun[find]
    #     P = Pun[find]
    #     for tind in range(len(ti)):
    #         for mk in range(32):
    #             s[mk] = A[mk][find, tind]
    #         # alphas = Mi * s # TEMPLATE
    #         W = np.matrix(np.diag(np.sum(np.array(P),1)**2))
    #         W /= np.max(W)
    #         # print(W.shape, P.shape, S.shape)
    #         alphas, _, _ ,_ = np.linalg.lstsq(W * P, W * S * s)
    #         # print(alphas)
    #
    #         #
    #         for sind in range(len(doas)):
    #             # alpha = synthbeamform(doas, sind, alphas, Nmax[kra]) # TEMPLATE
    #             pix = hp.ang2pix(4, doas[sind][0], doas[sind][1], nest=True)
    #             ysteer = S[pix, :]
    #             sr = np.reshape(np.array(ysteer * s), (1,))
    #             B[sind][find, tind] = alphas[sind,0]
    #             C[sind][find, tind] = sr[0]
    #
    #         # # CALCULATE TARGET AND INTERFERENCE
    #         # for sind2 in range(len(doas)):
    #         #     pix = hp.ang2pix(4, doas[sind2][0], doas[sind][1], nest=True)
    #         #     ysteer = S[pix, :]
    #         #     for mk2 in range(32):
    #         #         s[mk2] = srconly[sind2][mk2][find, tind]
    #         #     sr = np.reshape(np.array(ysteer * s), (1,))
    #         #     D[sind2][find, tind] = sr[0]
    #         #     alphassingle = Mi * s
    #         #     E[sind2][find, tind] = alphassingle[targetind,0]
    #
    # # CALCULATE TEMPLATE-BASED SS
    # out = []
    # for sind in range(len(doas)):
    #     out = sgen.postprocessoutput(B[sind], fs, NFFT, olap)
    #     print(type(out))
    #     out /= np.max(np.abs(out))
    #     sgen.wavwrite(out, fs, '/Users/huseyinhacihabiboglu/PycharmProjects/Separate/outtmp'+str(sind)+'.wav')
    #
    # # CALCULATE MAXIMUM DIRECTIVITY BEAMFORMING
    # out = []
    # for sind in range(len(doas)):
    #     out = sgen.postprocessoutput(C[sind], fs, NFFT, olap)
    #     print(type(out))
    #     out /= np.max(np.abs(out))
    #     sgen.wavwrite(out, fs, '/Users/huseyinhacihabiboglu/PycharmProjects/Separate/outbeam' + str(sind) + '.wav')
    #
    # # CALCULATE REFERENCES
    # out = []
    # for sind in range(len(doas)):
    #     out = sgen.postprocessoutput(D[sind], fs, NFFT, olap)
    #     print(type(out))
    #     out /= np.max(np.abs(out))
    #     sgen.wavwrite(out, fs, '/Users/huseyinhacihabiboglu/PycharmProjects/Separate/outref' + str(sind) + '.wav')
    #
    # # CALCULATE TARGET AND INTERFERENCES
    # out = []
    # for sind in range(len(doas)):
    #     if sind == targetind:
    #         out = sgen.postprocessoutput(E[sind], fs, NFFT, olap)
    #         print(type(out))
    #         # out /= np.max(np.abs(out))
    #         sgen.wavwrite(out, fs, '/Users/huseyinhacihabiboglu/PycharmProjects/Separate/target' + str(sind) + '.wav')
    #     else:
    #         out = sgen.postprocessoutput(E[sind], fs, NFFT, olap)
    #         print(type(out))
    #         # out /= np.max(np.abs(out))
    #         sgen.wavwrite(out, fs, '/Users/huseyinhacihabiboglu/PycharmProjects/Separate/interference' + str(sind) + '.wav')

        # Mi = getunmixingmatrix(Pi, YS, Bi, YQH)
        # M.append[Mi]


    # lev = 1
    # gran = 4
    # pls = legendrepulse(4, lev, np.pi/2, np.pi / 3)
    # pls = pls * (0.4 - .3j)
    # pls2 = legendrepulse(4, lev, np.pi / 2, -np.pi / 3)
    # pls2 = pls2 * (-0.9 + .7j)
    # pls += pls2
    # pls3 = legendrepulse(4, lev, np.pi / 4, 0)
    # pls3 = pls3 * (0.5 + .5j)
    # pls += pls3
    # D = legendredict(4, lev, gran)
    # print(D.dtype)
    # print(pls.dtype)
    # for ind in range(1):
    #     omp = OrthogonalMatchingPursuit(n_nonzero_coefs=5)
    #     omp.fit(D, np.real(pls))
    #     coefreal = omp.coef_
    #     indr = np.nonzero(coefreal)
    #     hp.mollview(coefreal, nest=True)
    #     plt.show()
    #     # Dr = D[:,ind]
    #     # omp.fit(D, np.imag(pls))
    #     # coefimag = omp.coef_

    # D = legendredict(4, 1, 1)
    # for ind in range(48):
    #     hp.mollview(D[:,ind],nest=True)
    #     plt.show()

    # 0. Set the simulation parameters
    Nm = 4
    NFFT = 2048
    olap = 1024 #512
    fs = 48000
    duration = 240000
    lev = 2
    gran = 2
    ra = 4.2e-2
    thc = np.pi/8
    Nbw = 10
    suppressdB = -10
    numsrc = 5
    kappa = 10 # 6 worked good

    offsetdB = -40

    offset = db2lin(offsetdB, 1.)

    sourceindex = 2


    # srcfl = ['speech-01-norm.wav', 'speech-02-norm.wav', 'speech-03-norm.wav' , 'speech-04-norm.wav']
    # srcind = [(0, 3, 2), (3, 0, 2), (6, 3, 2) ,(3, 6, 2)]

    srcfl = ['mozart_soprano-norm.wav', 'mozart_cello-norm.wav', 'mozart_violin1-norm.wav',  'mozart_violin2-norm.wav', 'mozart_clarinet-norm.wav', 'mozart_viola-norm.wav']
    # srcfl = ['mozart_violin1-norm.wav', 'mozart_violin2-norm.wav']
    # srcind = [(0, 3, 2), (2, 1, 2), (5, 1, 2), (5, 5, 2), (2, 5, 2), (6, 3, 2)] #Scenario 1
    srcind = [(0, 3, 3), (2, 2, 0), (5, 5, 1), (3, 0, 1), (3, 6, 0)] #Scenario 2
    # srcind = [(1, 2, 2), (2, 0, 2), (5, 1, 2), (5, 5, 2), (2, 6, 2), (1, 4, 2)] #Scenario 2
    # srcind = [(0, 3, 2), (2, 1, 2), (2, 5, 2), (3, 0, 2), (3, 6, 2)]


    cent = (3, 3, 2)
    doas = []

    sg = sgen.composescene3d(srcfl[:numsrc], srcind[:numsrc], duration)
    # sgen.wavwrite(np.sum(sg,0), 48000, '/Users/huseyinhacihabiboglu/PycharmProjects/Separate/mixture.wav')
    #
    ydict1 = getYnmtlookup('/Users/huseyinhacihabiboglu/PycharmProjects/Separate/ynmdict/ynm_N1_lvl2.pkl')
    ydict2 = getYnmtlookup('/Users/huseyinhacihabiboglu/PycharmProjects/Separate/ynmdict/ynm_N2_lvl2.pkl')
    ydict3 = getYnmtlookup('/Users/huseyinhacihabiboglu/PycharmProjects/Separate/ynmdict/ynm_N3_lvl2.pkl')
    ydict4 = getYnmtlookup('/Users/huseyinhacihabiboglu/PycharmProjects/Separate/ynmdict/ynm_N4_lvl2.pkl')

    ydict = [ydict1, ydict2, ydict3, ydict4]

    # Calculate dictionaries that contain Legendre pulses
    D1 = legendredict(1, lev, gran)
    D2 = legendredict(2, lev, gran)
    D3 = legendredict(3, lev, gran)
    D4 = legendredict(4, lev, gran)

    D = [D1, D2, D3, D4] # List containing the dictionaries for different maximum SHD orders

    # Obtain the STFTs of each of the microphone channels
    A, fi, ti = sgen.preprocessinput(sg, Nm, fs, NFFT, olap)
    shp = A[0].shape
    Omat = [] # To store our output
    Bmat = [] # to store beamformed output
    Rmat = [] # to store reference
    R = []
    for refind in range(numsrc):
        print(srcfl[refind])
        print(srcind[refind])
        sg = sgen.composescene3d([srcfl[refind]], [srcind[refind]], duration)
        Rm, fi, ti = sgen.preprocessinput(sg, Nm, fs, NFFT, olap)
        R.append(Rm)

    for ind in range(numsrc):
        th, ph = ind2doa(srcind[ind], cent)
        doas.append((th,ph))
        Omat.append(np.zeros(shp, dtype=complex))
        Bmat.append(np.zeros(shp, dtype=complex))
        Rmat.append(np.zeros(shp, dtype=complex))

    SR = []
    # Calculate the SR matrices for each different frequency range
    em32 = mic.EigenmikeEM32().returnAsStruct()
    thm = em32['thetas']
    phm = em32['phis']

    Nmax = range(1,5)
    s = np.matrix(np.zeros((32, 1), dtype=complex))
    rfs = np.matrix(np.zeros((32, 1), dtype=complex))

    for ind in range(numsrc):
        for tind in range(len(ti)):
            for m in range(32):
                Omat[ind][0, tind] += A[m][0,tind] / 32

    for find in range(1, NFFT / 2 + 1):
        print(find)
        k = 2 * np.pi * fi[find] / 340.
        ra = em32['radius']
        kra = k * ra
        if kra < 1:
            kra = 1
        elif kra < 2:
            kra = 2
        elif kra < 3:
            kra = 3
        else:
            kra =4
        Nd = Nmax[kra-1]
        # Pi, P = getPi(doas, lev, Nmax[kra])
        YS = getYS(Nd, lev)
        YQH = getYQH(thm, phm, Nd)
        Bi = getBi(k, ra, Nd)
        Smat = getsrmatrix(YS, Bi, YQH)

        srvec = []

        for src in range(numsrc):
            svec = steeringvector(doas[src], Nd)
            srvec.append(svec * Bi * YQH)

        for tind in range(len(ti)):
            for mk in range(32):
                s[mk] = A[mk][find, tind] # Signal vector

            y = Smat * s # This is the SR evaluated at a finite number of 12*2**(2*lev) points for the current TF bin
            inds, alphas = complexomp(D[kra-1], y, n_nonzero_terms = 10)
            # inda, alpa = nghalphas(doas[1], lev, gran, inds, alphas) # This is the one which works only on the neighbourhood of the given doa
            for isrc in range(numsrc):
                inda, alpa = maskedhalphas(doas[isrc], kappa, lev, gran, inds, alphas, offset) # WORKING
                # doasupp = list(doas)
                # doasupp.pop(isrc)
                # inda, alpa = maskedhalphas2(doas[isrc], doasupp, kappa, 9, lev, gran, inds, alphas, offset)
            # print(len(alpa))
                asrc = recompose(inda, lev, alpa, doas[isrc], Nd)
                Omat[isrc][find, tind] = asrc # Separated with complex OMP
                bfsrc = srvec[isrc] * s
                Bmat[isrc][find, tind] = bfsrc # Beamformed output
                for mk in range(32):
                    rfs[mk] = R[isrc][mk][find, tind]  # Signal vector
                rfsrc = srvec[isrc] * rfs
                Rmat[isrc][find, tind] = rfsrc  # Reference output

    for isrc in range(numsrc):
        outsnd = sgen.postprocessoutput(Omat[isrc], fs, NFFT, olap)
        outsnd /= np.max(np.abs(outsnd))
        sgen.wavwrite(outsnd, fs, '/Users/huseyinhacihabiboglu/PycharmProjects/Separate/Mozart_ns' + str(numsrc) + '/sep_source' + str(isrc) + '.wav')

        bfsnd = sgen.postprocessoutput(Bmat[isrc], fs, NFFT, olap)
        bfsnd /= np.max(np.abs(bfsnd))
        sgen.wavwrite(bfsnd, fs, '/Users/huseyinhacihabiboglu/PycharmProjects/Separate/Mozart_ns' + str(numsrc) + '/bf_source' + str(isrc) + '.wav')

        refsnd = sgen.postprocessoutput(Rmat[isrc], fs, NFFT, olap)
        refsnd /= np.max(np.abs(refsnd))
        sgen.wavwrite(refsnd, fs, '/Users/huseyinhacihabiboglu/PycharmProjects/Separate/Mozart_ns' + str(numsrc) + '/ref_source' + str(isrc) + '.wav')

    #     YS = getYS(Nmax[kra], lev)
    #     YQH = getYQH(thm, phm, Nmax[kra])
    #     Bi = getBi(k, ra, Nmax[kra])

    # # Spatially constrained dictionaries
    # D1, dt1, nrm1 = legendredict_sc(1, lev, gran, doas)
    # D2, dt2, nrm2 = legendredict_sc(2, lev, gran, doas)
    # D3, dt3, nrm3 = legendredict_sc(3, lev, gran, doas)
    # D4, dt4, nrm4 = legendredict_sc(4, lev, gran, doas)
    #
    # # D1 = legendredict_doas(1, lev, gran, doas)
    # # # print(np.matrix(D1).shape)
    # # D2 = legendredict_doas(2, lev, gran, doas)
    # # D3 = legendredict_doas(3, lev, gran, doas)
    # # D4 = legendredict_doas(4, lev, gran, doas)
    #
    # # D1, ipix = legendredictconstr(1, np.array([1, 0 ,0]), np.pi/4, lev, gran)
    # # D2, ipix = legendredictconstr(2, np.array([1, 0 ,0]), np.pi/4, lev, gran)
    # # D3, ipix = legendredictconstr(3, np.array([1, 0 ,0]), np.pi/4, lev, gran)
    # # D4, ipix = legendredictconstr(4, np.array([1, 0 ,0]), np.pi/4, lev, gran)
    # D = [D1, D2, D3, D4]
    # dt = [dt1, dt2, dt3, dt4]
    # nrm = [nrm1, nrm2, nrm3, nrm4]
    #
    # ###
    # # w1 = windowcreate(mu = doas[1], sigma = np.pi/2) # 4
    # # w2 = windowcreate(mu = doas[1], sigma = np.pi/4) # 8
    # # w3 = windowcreate(mu = doas[1], sigma = np.pi/6) # 12
    # # w4 = windowcreate(mu = doas[1], sigma = np.pi/8) # 16
    #
    # # Amplify target and suppress interference
    # # wampsupp = windowmask(doas[0], [doas[1]], np.pi/8, lev)
    # # wampsupp = bwwin(doas[0], thc, Nbw, 1, suppressdB)
    # # wampsupp = windowmask(doas[0], [doas[1]], np.pi/8, gran)
    # # win = [w1, w2, w3, w4]
    # ###
    # # walphas = windowmask(doas[0], [doas[1]], np.pi/8, gran)
    #
    # Pnm, fi, ti = sgen.preprocessinput(sg, Nm, fs, NFFT, olap)
    # shp = Pnm[0].shape
    # out = np.zeros(shp, dtype=complex)
    # start = tm.time()
    # print(shp[0], shp[1])
    #
    # srcindex = 3
    # numsrc = 4
    #
    # for find in range(shp[0]):
    #     print(find)
    #     Nmax = selectnmax(fi[find], ra)
    #     for tind in range(shp[1]):
    #         mvec = selectbinindx(find, tind, Pnm, Nmax)
    #         svec, _ = srford(mvec, fi[find], ra, lev, ydict[Nmax-1], Nmax) # Calculate SR
    #         # svec /= nrm[Nmax-1]
    #         inds, alphas = complexomp_nzt(D[Nmax-1], dt[Nmax-1], nrm[Nmax-1], svec, n_nonzero_terms = numsrc)
    #         # thi, phi = hp.pix2ang(2**gran, inds, nest=True)
    #         # print((thi/np.pi*180, phi/np.pi*180))
    #         # inds, alphas, Psinv = complexomp_doathr(D[Nmax-1], svec, Nmax, lev, doas, thr=0.01, ep=0.1, maxiter=30)
    #         # inds, alphas = complexomp_doathrw(D[Nmax-1], svec, Nmax, lev, wampsupp, doas, thr=0.01, ep=0.1, maxiter=20)
    #         # inds, alphas = selectalphas(inds, alphas, suppressgain=0., radius=np.pi/4, doai=doas[1:], interind=[1], gran=gran)
    #         # inds, alphas = suppressalphas(inds, alphas, suppressgain=0.01, radius=np.pi/4, doas=doas, indsrc=0, gran=gran) # OKish
    #
    #         ## THIS IS GOOD!
    #         # alphas = windowg(svec, Psinv, wampsupp) # Windowing works good!
    #         ## THIS IS GOOD!
    #
    #         # ### THIS IS NEW!
    #         # alphas = windowalphas(inds, alphas, 2, [1], walphas, gran)
    #         # ### THIS IS NEW!
    #         # alphas = windowalphas(doas, alphas, inds, wampsupp, gran)
    #         # alphas = alphamask(alphas, inds, lev, gran, doas[srcindex])
    #         # alphabm = beamformexact(alphas, inds, doas[srcindex][0], doas[srcindex][1], Nmax, gran)
    #         # alphabm = beamform(Psi, alphas, doas[0][0], doas[0][1], lev)
    #
    #         # alphabm = beamform(inds, alphas, doas[0][0], doas[0][1], Nmax, gran, doas)
    #         # inds, alphas = complexomp_thr(D[Nmax-1], svec, thr)
    #         inds, alphas = complexomp_doa(D[Nmax-1], svec, 6, Nmax, lev, doas)
    #         # if np.isnan(alphas[0]):
    #         #     alphas[0] = 0
    #         # if np.isnan(alphas[1]):
    #         #     alphas[1] = 0
    #         out[find,  tind] = alphas[srcindex]
    #         # if find == 6:
    #         #     doa = np.zeros(12 * 2**(2*gran))
    #         #     doa[inds]=np.array(alphas).reshape((len(alphas),))
    #         #     hp.mollview(np.abs(svec), nest=True)
    #         #     plt.show()
    #         #     hp.mollview(np.abs(doa), nest=True)
    #         #     plt.show()
    # stop = tm.time()
    #
    # # Post-process (e.g. ISTFT, normalisation) and write to sound file
    # outsnd = sgen.postprocessoutput(out, fs, NFFT, olap)
    # # outsnd /= np.max(np.abs(outsnd))
    # plt.plot(outsnd)
    # plt.show()
    # sgen.wavwrite(outsnd, fs, '/Users/huseyinhacihabiboglu/PycharmProjects/Separate/sepnrm' + str(srcindex) + '.wav')
    # print(stop - start)
    #
    # # for find in range(len(fi)):
    # #     for tind in range(len(ti)):
