import numpy as np
import scipy.fft as ft
import scipy.signal as sp
from scipy.special import sph_harm
from scipy.constants import speed_of_sound as c
from ambixutil import nm2acn
import legpulse as lg

def stftparams(fs = 48000, window='hann', nperseg=1024/2, noverlap=960/2, nfft=1024/2):
    assert sp.check_COLA(window, nperseg, noverlap)
    params = dict()
    params['window'] = window
    params['nperseg'] = nperseg
    params['noverlap'] = noverlap
    params['nfft'] = nfft
    params['fs'] = fs
    return params

def stft_sh(sharr, params):
    window = params['window']
    nperseg = params['nperseg']
    noverlap = params['noverlap']
    nfft = params['nfft']
    fs = params['fs']
    rw, cl = np.shape(sharr)
    stftlist = []
    for ind in range(cl):
        fr, tm, shstft = sp.stft(sharr[:,ind], fs=fs, window=window, nperseg=nperseg, noverlap=noverlap, nfft=nfft, return_onesided=False)
        stftlist.append(shstft)
    sz = np.shape(stftlist[0])
    Nmax = int(np.sqrt(cl)-1)
    return stftlist, fr, sz, Nmax

def getshvec(ind, jnd, stftlist, sz, orderlist):
    """
    ambix'ler shd'ye döndürülüp stft'si alındıktan sonra stftlist'te tutuluyor 16 x tbin x fbin olarak.
    getshvec ile istediğimi bin'in istediğimi orderdaki (Nbin) değerlerini vektör olarak döndürüyoruz.
    örn. Nbin=2,  tbin =24, fbin = 1299 olsun. 24e 1299 numaralı bindeki 16 adet değerin ilk 9 tanesini vc olarak
    döndürüyoruz
    """
    Nbin = orderlist[ind]
    assert (ind < sz[0]) and (jnd < sz[1])
    maxind = nm2acn(Nbin, Nbin)
    vc = np.zeros(maxind+1, dtype=complex)
    for nm in range(maxind+1):
        vc[nm] = stftlist[nm][ind,jnd]
    return vc


def fft_sh(sharr, fs):
    rw, cl = np.shape(sharr)
    fftlist = []
    for ind in range(cl):
        fr_sh = ft.fft(sharr[:,ind])
        nfft = np.shape(fr_sh)[0]
        fftlist.append(fr_sh[:int(nfft/2)])
    fr = ft.fftfreq(nfft) * fs
    print("nfft/2", int(nfft/2))
    return fftlist, fr

def getshvec_fft(ind,fftlist,orderlist):
    Nbin = orderlist[ind]
    sz = np.shape(fftlist[0])[0]/2
    assert (ind < sz)
    maxind = nm2acn(Nbin, Nbin)
    vc = np.zeros(maxind+1, dtype=complex)
    for nm in range(maxind+1):
        vc[nm] = fftlist[nm][ind]
    return vc


def istft_sh(stftlist, params):
    pass

def ynmmat(gridorder, Nmax):
    _, angs, npix = lg.grid(gridorder)
    ynmlist = []
    for ind in range(npix):
        ynm = np.zeros(((Nmax+1)**2,1), dtype=complex)
        for n in range(Nmax):
            for m in range(-n, n+1):
                jnd = nm2acn(n,m)
                ynm[jnd] = sph_harm(m, n, angs[ind][1], angs[ind][0])
        ynmlist.append(ynm.copy())
    return ynmlist

def srf(shvec, ipix, ynmlist):
    yvec = ynmlist[ipix]
    return np.dot(shvec, yvec[:np.shape(shvec)[0]])
    #return np.dot(shvec, yvec[:np.shape(shvec)[1]])

def srfmap(shvec, ynmlist):
    srfmap = []
    npix = len(ynmlist)
    for ipix in range(npix):
        srfmap.append(srf(shvec, ipix, ynmlist))
    return np.array(srfmap)

def srf_single(shvec, ipix, ynmlist):
    yvec = ynmlist[ipix]
    #return np.dot(shvec, yvec[:np.shape(shvec)[0]])
    return np.dot(shvec, yvec[:np.shape(shvec)[1]])

def srfmap_single(shvec, ynmlist):
    srfmap = []
    npix = len(ynmlist)
    for ipix in range(npix):
        srfmap.append(srf_single(shvec, ipix, ynmlist))
    return np.array(srfmap)

def srp(shvec, ipix, ynmlist):
    s = srf(shvec, ipix, ynmlist)
    return np.abs(s)

def srpmap(shvec, npix, ynmlist):
    s = srfmap(shvec, npix, ynmlist)
    return np.abs(s)**2

def ordervec(fr, ra, Nmax):
    fcnt = len(fr)
    kra = np.abs(2 * np.pi * fr / c * ra)
    orderlist = []
    for find in range(fcnt):
        krai = kra[find]
        orderlist.append(int(clamp(np.round(krai), Nmax)))
    return orderlist

def clamp(a, amax):
    if a > amax:
        return amax
    else:
        return a

def plotsrp(srp):
    pass

def pltsrf(srf, flag=0):
    pass

if __name__ == '__main__':
    print(clamp(4, 3))