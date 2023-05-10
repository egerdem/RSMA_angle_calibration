from scipy.signal import fftconvolve
from scipy.signal import oaconvolve
import scipy.signal as sg
import numpy as np
from numpy import matlib
from scipy.io import wavfile
import pwdec as pwd
import ambixutil as autil
import ompseparate as omp
import healpy as hp
from scipy.signal import hilbert
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, windows
import pickle as pckl
from scipy.fft import rfft
from itertools import combinations


def corralign(ch1, ch2):
    crr = sg.correlate(ch1, ch2, mode="full", method="fft")
    lags = sg.correlation_lags(ch1.size, ch2.size, mode="full")
    lag = lags[np.argmax(crr)]
    ind = np.argmax(crr)
    delay = len(crr) / 2 - ind
    # plt.plot(crr)
    # plt.show()
    # print(lag)
    return lag


def toa(mics, src, fs, c=341.):
    toa_list = []
    key, mic_locs = zip(*mics.items())
    for i in range(len(mic_locs)):
        toa_list.append(int(np.linalg.norm(src - mic_locs[i]) * fs / c))
    print("toa_sample =", toa_list)
    return (toa_list)


def shift(raw0, shift_sample):
    if shift_sample < 0:
        assert "error"
    else:
        # print("shifted %d" %shift_sample )
        pre = np.pad(raw0, ((shift_sample, 0), (0,0)), constant_values=0)
    return(pre)


def shift_list(rawlist, samplelist):
    # raw_list =  7x512 (channel no x signal length)
    shifted_list = []
    for ind in range(len(samplelist)):
        shifted_list.append(shift(rawlist[ind], samplelist[ind]))
    return(shifted_list)


def raw2ambix(ref_data, irdir):
    # rate1, raw = wavfile.read(filepos, mmap=False)
    # irdir = '/Users/orhunsparg/PycharmProjects/somp-interpolation/data/A2B-Zylia-3E-Jul2020.wav'
    # irdir = '/Users/ege/PycharmProjects/somp-interpolation/data/A2B-Zylia-3E-Jul2020.wav'
    raw = ref_data
    rate2, eir = wavfile.read(irdir, mmap=False)
    # assert rate1 == rate2 # Make sure that the sampling rates are the same
    rowd, cold = np.shape(raw)  # Data
    rowi, coli = np.shape(eir)  # Impulse (FARINA FILTERS)
    irsize = int(rowi / (cold))  # Last channel is timecode, we will not process it here
    ambix = np.zeros((rowd + irsize - 1, coli), dtype=float)
    for ind in range(coli):
        # ir = eir[:,ind]
        ir = eir[:, ind] / (2. ** 31)  # Raw recordings are 32-BIT DO NOT DELETE
        for jnd in range(cold):
            sig = raw[:, jnd] #/ (2. ** 31)
            ire = ir[jnd * irsize: (jnd + 1) * irsize]
            eqsig = fftconvolve(sig, ire, mode='full')
            ambix[:, ind] += eqsig
    return raw, ambix

def rotation_azimuth(shd, phi):
    var = len(shd)
    #var = min(rw,cl)
    N = int(np.sqrt(var)-1)
    mlist =[]
    for n in range(N+1):
        for m in range(-n,n+1):
            mlist.append(np.exp(1j*m*phi))
    marr = np.array(mlist)
    rotatedshd = shd*marr
    return rotatedshd

def angle_between_vecs_z(vec1,vec2):
    #assuming vec1 and vec2 are in horizontal plane
    vec1 /= np.linalg.norm(vec1)
    vec2 /= np.linalg.norm(vec2)
    vecv = np.cross(vec2,vec1)
    vecz = np.array([0,0,1.])
    angle = np.arcsin(np.dot(vecv, vecz))
    return angle

def calc_dict(lev,gran):
    ##OMP##
    lev = lev
    gran = gran
    D1 = omp.legendredict(1, lev, gran)
    D2 = omp.legendredict(2, lev, gran)
    D3 = omp.legendredict(3, lev, gran)
    D4 = omp.legendredict(4, lev, gran)
    D = [D1, D2, D3, D4]
    return D

def rotate2D(theta, vec):
    vec2d = np.array([vec[0], vec[1]])
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    rotvec = R @ vec2d
    rotvec = np.array([rotvec[0], rotvec[1], vec[2]])
    return rotvec

def rotation_calibration(micpos, srcpos, freq, shifted_raw, D, lev,gran, fs, irdir):
    Nmax = 3
    ra = 5.5e-2 # ra for Zylia
    abx = raw2ambix(shifted_raw, irdir)
    sharr = autil.ambix2sh(abx)
    #params = pwd.stftparams()
    #shs, fr, sz, Nmax = pwd.stft_sh(sharr, params)
    shs, fr = pwd.fft_sh(sharr, fs)

    frdiff = abs(fr - freq)
    fr_ind = np.where(frdiff==np.min(frdiff))[0][0]

    freq = fr[fr_ind]
    kra = int(np.floor(2* np.pi * freq/341. * ra)) ### or round()?
    print("kra", kra)
    kra = pwd.clamp(kra,2)
    print("freq", freq)
    print("fr_ind", fr_ind)

    orderlist = pwd.ordervec(fr, ra , Nmax)
    a = pwd.getshvec_fft(fr_ind, shs, orderlist)
    ynml = pwd.ynmmat(lev, Nmax)
    s = pwd.srfmap(a, ynml)
    #hp.mollview(np.abs(s.reshape((12*2**(2*lev),))), nest=True)
    #plt.show()
    inds, alphas = omp.complexomp(D[kra - 1], s, n_nonzero_terms=1)

    nside = hp.npix2nside(12*2**(2*gran))
    x1, y1, z1 = hp.pix2vec(nside, inds, nest=True)

    vec = [x1[0], y1[0], 0]  # horizontal assumption for calibration
    print("vec", vec)
    vectrue = -(micpos - srcpos) / np.linalg.norm(micpos - srcpos)
    print("vectru", vectrue)
    # HORIZONTAL SOURCE AND MIC ASSUMPTION
    vectrue[2] = 0
    angle_diff = angle_between_vecs_z(vectrue, vec)

    return angle_diff

def filter_audio(irdir,filedirs):
    # takes filter and raw audio and returns both ambix and the raw of the filtered audio
    rawlist = []
    ambix_list = []
    for file in filedirs:
        rate1, abx1, raw1 = autil.raw2ambix(file, irdir)
        rawlist.append(raw1)
        ambix_list.append(np.transpose(abx1))
    return rawlist, ambix_list

def get_audio_convolved(irdir, filedirs, insigdir):
    rawlist = []
    ambix_list = []
    rate, insig = wavfile.read(insigdir, mmap=False)
    row_sig= len(insig)
    #plt.plot(insig)
    #plt.show()
    for file in filedirs:
        rate, abx, raw = autil.raw2ambix(file, irdir)
        row, col = np.shape(raw)
        conv_signal=[]
        for ind in range(col):
            #plt.plot(raw[:,ind])
            #plt.plot(insig)
            #plt.show()
            conv_sig_temp = oaconvolve(raw[:,ind],insig, mode='full')
            conv_signal.append(conv_sig_temp)
            #plt.plot(conv_sig_temp)
            #plt.show()
        conv_signal_org = np.array(conv_signal)
        conv_signal = conv_signal_org.T#reshape(row_sig+row-1, col)
        abx_conv = raw2ambix(conv_signal, irdir)
        rawlist.append(conv_signal)
        ambix_list.append(np.transpose(abx_conv))
    return rawlist, ambix_list

def get_audio_convolved_raw(irdir, rawsigs, insigdir):
    rawlist = []
    ambix_list = []
    rate, insig = wavfile.read(insigdir, mmap=False)
    row_sig= len(insig)
    #plt.plot(insig)
    #plt.show()
    for rawsig in rawsigs:
        raw, abx = raw2ambix(rawsig, irdir)
        row, col = np.shape(raw)
        conv_signal = []
        for ind in range(col):
            #plt.plot(raw[:,ind])
            #plt.plot(insig)
            #plt.show()
            conv_sig_temp = oaconvolve(insig, raw[:, ind], mode='full')
            conv_signal.append(conv_sig_temp)
            #plt.plot(conv_sig_temp)
            #plt.show()
        conv_signal_org = np.array(conv_signal)
        conv_signal = conv_signal_org.T#reshape(row_sig+row-1, col)
        _, abx_conv = raw2ambix(conv_signal, irdir)
        rawlist.append(conv_signal)
        #ambix_list.append(np.transpose(abx_conv))
        ambix_list.append(abx_conv)
    return rawlist, ambix_list

def convolve_sig(irdir, filedirs, freq, t, fs):
    rawlist = []
    ambix_list = []
    cosine  = np.cos(2*np.pi*freq*np.linspace(0,t,int(t*fs)))
    for file in filedirs:
        rate, abx, raw = autil.raw2ambix(file, irdir)
        row, col = np.shape(raw)
        print("row", row, col)
        conv_signal=[]
        for ind in range(col):
            #plt.plot(raw[:,ind])
            #plt.plot(insig)
            #plt.show()
            conv_sig_temp = oaconvolve(raw[:,ind],cosine, mode='full')
            conv_signal.append(conv_sig_temp)
            #plt.plot(conv_sig_temp)
            #plt.show()
        conv_signal_org = np.array(conv_signal)
        conv_signal = conv_signal_org.T#reshape(row_sig+row-1, col)
        abx_conv = raw2ambix(conv_signal, irdir)
        rawlist.append(conv_signal)
        ambix_list.append(abx_conv)
    return rawlist, ambix_list

def select_window(type, size):

    return windows.get_window(type,size)

def crop_directpath(shifted_raws, window):

    analytic_signal = hilbert(shifted_raws[0][:,0])
    amplitude_envelope = np.abs(analytic_signal)
    peaks, props = find_peaks(amplitude_envelope, height=0,distance=50)

    direct_peak_ind = np.where(amplitude_envelope==np.max(props["peak_heights"]))
    rw = np.shape(window)
    rw = rw[0]
    cropped_raws = []
    # print("hann window of size =", rw)
    if rw%2 == 1:
        assert ValueError("window can only be even size")
    for raw in shifted_raws:
        IRcrop = raw[direct_peak_ind[0][0] - int(rw/2): direct_peak_ind[0][0] + int(rw/2),:] * matlib.repmat(window,19,1).T
        cropped_raws.append(IRcrop)
    #plt.plot(direct_peak_ind, amplitude_envelope[direct_peak_ind], "x")
    #plt.plot(shifted_raws[0][:, 0])
    #plt.plot(amplitude_envelope)
    #plt.show()
    return cropped_raws


def crosscor_delays(ambix_list):
    delay_list = np.zeros(len(ambix_list))
    for i in range(len(ambix_list) - 1):
        delay_list[i + 1] = int(corralign(ambix_list[0][0, :] / (2 ** 31), ambix_list[i + 1][0, :] / (2 ** 31)))
        # print(delay_list)
        # print("i, i+1 ", 1, i + 2)

    #rawlist = np.transpose(rawlist[0])
    samplelist = np.array(delay_list) - np.min(delay_list)
    crosscorr_del = samplelist.astype(int)
    #shifted_raws = shift_list(rawlist, samplelist_neworg)
    return crosscorr_del


def conv_signal_time_align(irdir, filedirs, insigdir):
    rawlist_conv, ambixlist_conv = get_audio_convolved(irdir, filedirs, insigdir)
    rawlist, ambixlist = get_audio(irdir, filedirs)

    #align_delay_arr = crosscor_delays(ambixlist_conv)
    align_delay_arr = crosscor_delays(ambixlist)
    conv_shifted_raws = shift_list(rawlist_conv, align_delay_arr)

    return conv_shifted_raws, align_delay_arr

def conv_raw(irdir, rawsigs, insigdir):
    rawlist_conv, ambixlist_conv = get_audio_convolved_raw(irdir, rawsigs, insigdir)
    #rawlist, ambixlist = get_audio(irdir, filedirs)

    #align_delay_arr = crosscor_delays(ambixlist_conv)
    #align_delay_arr = crosscor_delays(ambixlist)
    #conv_shifted_raws = shift_list(rawlist_conv, align_delay_arr)

    return rawlist_conv, ambixlist_conv


def time_align(irdir, filedirs):
    """
    :param irdir: Zylia Ambisonics Farina IR path
    :param filedirs: Impulse responses directory of calibration recordings
    :return: Shifted impulse responses
    """
    rawlist, ambixlist = filter_audio(irdir, filedirs) # raw audio is filtered and returned both as rawlist and the ambixlist
    #fft_raw = rfft(ambixlist[2][0])
    #plt.plot(20*np.log10(fft_raw))
    #plt.show()
    crosscorr_del = crosscor_delays(ambixlist)
    shifted_raws = shift_list(rawlist, crosscorr_del)
    return shifted_raws, crosscorr_del, ambixlist

def align_recording_delays(rawlist, filteredambixlist):
    rawlist = np.array(rawlist)
    #fft_raw = rfft(ambixlist[2][0])
    #plt.plot(20*np.log10(fft_raw))
    #plt.show()
    crosscorr_del = crosscor_delays(filteredambixlist)
    # shifted_filtered_raws = shift_list(rawlist, crosscorr_del)
    shifted_filtered_raws = shift_list_multich(rawlist, crosscorr_del)
    return crosscorr_del, np.array(shifted_filtered_raws)

def shift_multi(raw_19, shift_sample, size):

    if shift_sample < 0:
        assert "error"
    else:
        numData = raw_19.shape[0]
        numChan = raw_19.shape[1]
        # print("data", numData)
        # print("chan", numChan)
        pre = np.zeros((numData+size, numChan))
        datasize = numData+shift_sample
        for ind in range(numChan):
        # print("shifted %d" %shift_sample )
            pre[:datasize,ind] = np.pad(raw_19[:,ind], (shift_sample, 0), constant_values=0)
            #pre = pre[:,ind].reshape(-1,1).flatten()
            #plt.plot(awut)
        #plt.show()
        #plt.figure()
    return pre
def shift_list_multich(rawlist_arr, samplelist):
    shifted_list_multi = []
    numMic = rawlist_arr.shape[0]
    numData = rawlist_arr.shape[1]
    numChan = rawlist_arr.shape[2]
    size = np.max(samplelist)
    print("samplelist: ", samplelist)
    for ind in range(len(samplelist)):
        shifted_list_multi.append(shift_multi(rawlist_arr[ind, :, :], samplelist[ind], size))
    return(shifted_list_multi)

def shift_list_multi(rawlist_arr, samplelist):
    # raw_list =  7x512 (channel no x signal length)
    shifted_list = []
    size = np.max(samplelist)
    for ind in range(len(samplelist)):
        shifted_list.append(shift_multi(rawlist_arr[ind, :, :], samplelist[ind], size))
    return(shifted_list)
def align_toa_multi(raws, mics, src, fs):
    toa_list = toa(mics, src, fs, c=341.)
    print("toa delays:", toa_list)
    toa_aligned_wavs = shift_list_multi(raws, toa_list)
    return(toa_aligned_wavs)
def select_dict(filepath, lev=3, gran=3):
    filename = filepath + '/D' + str(lev) + str(gran) + '.pkl'
    file = open(filename, "rb")
    D = pckl.load(file)
    file.close()
    return D


def calibration_angle(dict_path, mic_pos, src_pos, freqs, cropped_raw, fs, lev=3, gran=3):
    D = select_dict(dict_path, lev, gran)
    angle_avg = 0
    for freq in freqs:
        angle_diff = rotation_calibration(mic_pos, src_pos, freq, cropped_raw, D, lev, gran, fs)
        angle_avg += angle_diff
    angle_avg /= len(freqs)

    #print("anglediff1", np.rad2deg(angle_diff1))
    #print("anglediff2",  np.rad2deg(angle_diff2))
    return angle_avg

def rot_calib_SRF_fft(angle_diff, freq, raw, fs, lev, Nmax=3):
    ra = 5.5e-2 # Zylia ra

    abx = raw2ambix(raw, irdir)
    sharr = autil.ambix2sh(abx)

    shs, fr = pwd.fft_sh(sharr, fs)
    frdiff = abs(fr - freq)
    fr_ind = np.where(frdiff == np.min(frdiff))[0][0]

    freq = fr[fr_ind]
    kra = int(np.floor(2 * np.pi * freq / 341. * ra)) # or round()
    print("kra", kra)
    kra = pwd.clamp(kra, 2)

    orderlist = pwd.ordervec(fr, ra, Nmax)
    a = pwd.getshvec_fft(fr_ind, shs, orderlist)
    a = rotation_azimuth(a,angle_diff)
    ynml = pwd.ynmmat(lev, Nmax)
    srf_fft = pwd.srfmap(a, ynml)
    return srf_fft, kra

def tfbin_selection(conv_list, fftsize, irdir, method_flag="multiply"):
    sh_tfbins_list = []
    col_min = np.inf
    for raw in conv_list:
        abx = raw2ambix(raw, irdir)
        sharr = autil.ambix2sh(abx)
        params = pwd.stftparams()
        shs, fr, sz, Nmax = pwd.stft_sh(sharr, params)
        row, col = np.shape(shs[0])
        if col < col_min:
            col_min = col
        sh_tfbins_list.append(shs[0])
    sh_tfbins = np.ones((row, col_min))

    if method_flag == "multiply":
        for shs in sh_tfbins_list:
            shs = shs[:,:col_min]
            sh_tfbins *= np.abs(shs)

    if method_flag == "sum":
        for shs in sh_tfbins_list:
            shs = shs[:,:col_min]
            sh_tfbins += np.abs(shs)
    #plt.imshow(sh_tfbins)
    #plt.show()
    maxlist = max(map(max,sh_tfbins))
    print(maxlist)
    max_ind = np.where(sh_tfbins == maxlist)
    max_ind_freq = max_ind[0][0]
    max_ind_time = max_ind[1][0]
    if max_ind_freq >= (fftsize/2):
        max_ind_freq = fftsize-max_ind_freq
    peak = (max_ind_freq, max_ind_time)
    #peaks, props = find_peaks(sh_tfbins, height=0, distance=50)
    return peak


def ambix_srf_stft(freqq, abx, fs, peak_tf, lev, Nmax=3):
    ra = 5.5e-2 # Zylia ra
    tfbins = peak_tf

    #abx = clb.raw2ambix(raw)
    sharr = autil.ambix2sh(abx)
    params = pwd.stftparams()

    shs, fr, sz, Nmax = pwd.stft_sh(sharr, params)
    #shs, fr = pwd.fft_sh(sharr, fs)
    plt.imshow(abs(shs[0]))
    plt.show()
    #shs, fr = pwd.fft_sh(sharr, fs)
    frdiff = abs(fr - freqq)
    fr_ind = np.where(frdiff == np.min(frdiff))[0][0]

    freq = fr[fr_ind]
    kra = int(np.floor(2 * np.pi * freq / 341. * ra)) # or round()
    print("kra", kra)
    kra = pwd.clamp(kra, 2)

    orderlist = pwd.ordervec(fr, ra, Nmax)
    #a = pwd.getshvec(fr_ind, shs, orderlist, sz, orderlist)
    a = pwd.getshvec(tfbins[0], tfbins[1], shs, sz, orderlist)
    #a = pwd.getshvec_fft(fr_ind, shs, orderlist)
    ynml = pwd.ynmmat(lev, Nmax)
    srf_stft = pwd.srfmap(a, ynml)
    hp.mollview(np.abs(srf_stft.reshape((12*2**(2*lev),))), nest=True)
    plt.show()

    return srf_stft, kra


def rot_calib_SRF_STFT(angle_diff, freqq, raw, fs, peak_tf, lev, irdir, Nmax=3):
    ra = 5.5e-2 # Zylia ra
    #tfbins = (17, 4329)
    #tfbins = peak_tf #(30, 5214)
    #tfbins = (30, 5214)
    tfbins = (19, 442)

    abx = raw2ambix(raw, irdir)
    sharr = autil.ambix2sh(abx)
    params = pwd.stftparams()
    shs, fr, sz, Nmax = pwd.stft_sh(sharr, params)
    plt.imshow(abs(shs[0]))
    plt.show()
    #shs, fr = pwd.fft_sh(sharr, fs)
    frdiff = abs(fr - freqq)
    fr_ind = np.where(frdiff == np.min(frdiff))[0][0]

    freq = fr[fr_ind]
    kra = int(np.floor(2 * np.pi * freq / 341. * ra)) # or round()
    print("kra", kra)
    kra = pwd.clamp(kra, 2)

    orderlist = pwd.ordervec(fr, ra, Nmax)
    #a = pwd.getshvec(fr_ind, shs, orderlist, sz, orderlist)
    a = pwd.getshvec(tfbins[0],tfbins[1], shs, sz, orderlist)
    a = rotation_azimuth(a,-angle_diff)
    ynml = pwd.ynmmat(lev, Nmax)
    srf_stft = pwd.srfmap(a, ynml)
    hp.mollview(np.abs(srf_stft.reshape((12*2**(2*lev),))), nest=True)
    plt.show()

    return srf_stft, kra


def sparsedecomp_to_vectors(srf, kra, lev, gran, n_terms):
    D = select_dict('.', lev=lev, gran=gran)
    inds, alphas = omp.complexomp(D[kra - 1], srf, n_nonzero_terms=n_terms)
    nside = hp.npix2nside(12 * 2 ** (2 * gran))
    vecs = []
    for ind in inds:
        x, y, z = hp.pix2vec(nside, ind, nest=True)
        vec = [x, y, z]
        vecs.append(vec)
    return vecs, alphas

'''
def calculate_vecslistlist(vecslist, micposlist):
    
    pass

def calculate_linelistlist(vecslist, micposlist):
    miccount = len(micposlist)
    linelistlist = []
    for micind in range(miccount):
        micpos = micposlist[micind]
        veclist = vecslist[micind]
        linelist = []
        for vec in veclist:
            line = geo.Line(micpos, np.array(vec))
            linelist.append(line)
        linelistlist.append(linelist.copy())
    return linelistlist

def vectors_intersect(vecslist, micposlist):
    pass



    line1 = geo.Line(micpos1, np.array(vec1))  # line(point,vec)
    line2 = geo.Line(micpos2, np.array(vec2))
    line3 = geo.Line(micpos3, np.array(vec3))
    ptself, ptother, midpt, dist = line1.midptdistfromline(line2)
    comb = combinations(range(1, miccount + 1), 2)
    for i, j in list(comb):
        print(i,j)
'''