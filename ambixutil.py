import numpy as np
from scipy.io import wavfile
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt
# from pyo import savefile, Server, SndTable
import scipy.fft as fft
import pickle
#import dirpathimpulse as dirpath

def raw2ambix(filepos, irpos):
    rate1, raw = wavfile.read(filepos, mmap=False)
    rate2, eir = wavfile.read(irpos, mmap=False)
    assert rate1 == rate2 # Make sure that the sampling rates are the same
    rowd, cold = np.shape(raw) # Data
    rowi, coli = np.shape(eir) # Impulse (FARINA FILTERS)
    irsize = int(rowi / (cold)) # Last channel is timecode, we will not process it here
    ambix = np.zeros((rowd + irsize - 1, coli), dtype=float)
    for ind in range(coli):
        #ir = eir[:,ind]
        ir = eir[:,ind]/(2.**32) # Raw recordings are 32-BIT DO NOT DELETE
        for jnd in range(cold):
            sig = raw[:,jnd]
            ire = ir[jnd * irsize : (jnd+1) * irsize]
            eqsig = fftconvolve(sig, ire, mode='full')
            ambix[:,ind] += eqsig
    return rate1, ambix, raw


def ambixread(path):
    s = Server().boot()
    s.start()
    t = SndTable(path)
    samples = t.getTable(all = True)
    s.stop()
    del(s)
    ambarr = list2array(samples)
    return ambarr


def list2array(listoflists):
    ch = len(listoflists)
    sz = len(listoflists[0])
    arr = np.zeros((sz, ch))
    for ind in range(ch):
        arr[:, ind] = np.array(listoflists[ind])
    return arr


def nm2acn(n, m):
    # Convert (n,m) indexing to ACN indexing
    return n**2 + n + m


def ambix2sh(ambixchans):
    rw, cl = np.shape(ambixchans)
    # print("ambix2sh")
    # print(rw)
    # print(cl)
    shchannels = np.zeros((rw, cl), dtype=complex)
    N = int(np.sqrt(cl)-1)

    for n in range(N+1):
        ngain = np.sqrt(2 * n + 1)
        sqrt2 = np.sqrt(2)
        for m in range(-n, n + 1):
            chanind1 = nm2acn(n, m)
            chanind2 = nm2acn(n, -m)
            if m < 0:
                shchannels[:, chanind1] = ngain / sqrt2 * (ambixchans[:,chanind2] + 1j * ambixchans[:,chanind1])
            elif m > 0:
                shchannels[:, chanind1] = (-1)**m * ngain / sqrt2 * (ambixchans[:, chanind1] - 1j * ambixchans[:, chanind2])
                #shchannels[:, chanind1] = ngain / sqrt2 * (ambixchans[:, chanind1] - 1j * ambixchans[:, chanind2])
            else: # when m=0
                shchannels[:, chanind1] = ngain * ambixchans[:, chanind1]

    return shchannels


def fdambix(ambixchans,nfft):
    #Ambix to Frequency domain translation
    rw, cl = np.shape(ambixchans)
    fda = []
    for ind in range(cl):
        fda.append(fft.rfft(ambixchans[:, ind], n=nfft))
    return fda


def fdlist2array(fda):
    cl = len(fda)
    rw = np.shape(fda[0])[0]
    fdarray = np.zeros((rw, cl), dtype=complex)
    for ind in range(cl):
        fdarray[:, ind] = fda[ind]
    return fdarray


def sh2ambix(sharray):
    # TODO: Implement this!
    pass


def saveambix(ambarray, rate, path, chans=16):
    # https://github.com/ruda/caf/tree/master/caf
    samples = listoflists(ambarray)
    savefile(samples, path, rate, chans, fileformat=0, sampletype=3)
    #fileformat=O for .wav
    #fileformat=3 for .caf
    pass

def listoflists(array):
    rw, cl = np.shape(array)

    samples = []
    ls = list(np.transpose(array))
    for ch in range(cl):
        samples.append(list(ls[ch]))
    return samples

if __name__=='__main__':
    from pyo import savefile, Server, SndTable

    irdir = './data/A2B-Zylia-3E-Jul2020.wav'
    #filedir = '/Users/huseyinhacihabiboglu/PycharmProjects/somp-interpolation/data/ZM-1_20210524T194705.918+0300_M1.wav'
    #filedir = './data/lab_IRs/double/ir0.wav'
    filedir1 = './data/directpath/rotation/ir0_norm_ang3.wav'
    filedir2 = './data/directpath/rotation/ir1_norm_ang3.wav'
    rate, abx1 = raw2ambix(filedir1, irdir)
    rate, abx2 = raw2ambix(filedir2, irdir)
    samples = listoflists(abx1)
    #outfilepath1 = './data/out_direct1_30cm.wav'
    #outfilepath2 = './data/out_direct2_30cm.wav'
    #saveambix(abx, rate, '/Users/huseyinhacihabiboglu/PycharmProjects/somp-interpolation/data/try.caf')
    #saveambix(abx1, rate, outfilepath1)
    #saveambix(abx2, rate, outfilepath2)
    plt.plot(abx1[:,0])
    plt.show()
    hop=64
    #drpth, numchan =  dirpath.directpath(outfilepath, hop)
    #drpth, numchan =  raw2ambix(filedir, irdir)
    fd1 = fdambix(abx1, 1024)
    fd2 = fdambix(abx2, 1024)
    fda1 = fdlist2array(fd1)
    fda2 = fdlist2array(fd2)
    fdsh1 = ambix2sh(fda1)
    fdsh2 = ambix2sh(fda2)
    plt.plot(20*np.log10(np.abs(fd1[0])))
    plt.show()
    #sh = ambix2sh(abx)
    direct_30cm1 = open('direct_ang3_mic1', "wb")
    pickle.dump(fdsh1, direct_30cm1)
    direct_30cm1.close()
    direct_30cm2 = open('direct_ang3_mic2', "wb")
    pickle.dump(fdsh2, direct_30cm2)
    direct_30cm2.close()
    #plt.plot(np.real(sh[:,1]))
    #plt.plot(abx[:,1])
    #plt.show()

    # rate, raw = wavfile.read(filedir, mmap=False)
    # plt.plot(abx[:,1])
    # plt.plot(raw[:,0])
    # plt.show()

