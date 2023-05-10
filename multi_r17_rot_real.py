import json
import Microphone as mc
from scipy.signal import fftconvolve
import scipy.signal as sg
import scipy.fft as fft
import scipy.special as sp
from scipy.special import spherical_jn, sph_harm
import numpy as np
import recursion_r7 as recur
from numpy import matlib as mtlb
from scipy import linalg as LA
import scipy.fft as fft
from scipy.optimize import minimize
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import combinations
import pickle
from timeit import default_timer as timer
from scipy.optimize import basinhopping, brute, differential_evolution
import matplotlib
from datetime import datetime
import calibration as clb
import glob
import os
import ambixutil as amb
from scipy.io.wavfile import write
from scipy.io import wavfile

matplotlib.rc('xtick', labelsize=18)
matplotlib.rc('ytick', labelsize=18)

""" date: 22 haziran 22
    author: ege
    name: multi_r16.py : mds transformation deniyorum, load pickle ile md.py çıktılarını kullanarak
    
    27.09.22 fonksiyonları silip ortak dosyadan çağırma
    """

def cart2sphr(xyz):  # Ege'
    """ converting a multiple row vector matrix from cartesian to spherical coordinates """
    c = np.zeros((xyz.shape[0], 2))
    r = []
    tt = []
    for i in range(xyz.shape[0]):
        x = xyz[i][0]
        y = xyz[i][1]
        z = xyz[i][2]

        rad = np.sqrt(x * x + y * y + z * z)
        tt.append(rad)
        theta = np.arccos(z / rad)
        phi = np.arctan2(y, x)
        c[i][0] = theta
        c[i][1] = phi
        r.append(rad)

    return [c, np.array(r)]


def cart2sph_single(xyz):  # (ege)
    """ converting a single row vector from cartesian to spherical coordinates """
    # takes list xyz (single coord)
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]

    r = np.sqrt(x * x + y * y + z * z)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    return (r, theta, phi)


def cart2sph(x, y, z):
    """

    :param x: x-plane coordinate
    :param y: y-plane coordinate
    :param z: z-plane coordinate
    :return: Spherical coordinates (r, th, ph)
    """
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    th = np.arccos(z / r)
    ph = np.arctan2(y, x)
    return r, th, ph


def cart2sphr_sparg(xyz):
    """ converting a multiple row vector matrix from cartesian to spherical coordinates """
    c = np.zeros((xyz.shape[0], 2))
    r = []
    tt = []
    for i in range(xyz.shape[0]):
        x = xyz[i][0]
        y = xyz[i][1]
        z = xyz[i][2]

        rad = np.sqrt(x * x + y * y + z * z)
        tt.append(rad)
        theta = np.arccos(z / rad)
        phi = np.arctan2(y, x)
        c[i][0] = theta
        c[i][1] = phi
        r.append(rad)
    return (np.array(r), c[:, 0], c[:, 1])
    # return [c, np.array(r)]


def sph2cart(r, th, ph):
    x = r * np.sin(th) * np.cos(ph)
    y = r * np.sin(th) * np.sin(ph)
    z = r * np.cos(th)
    u = np.array([x, y, z])
    return u


def mic_sub(mic_p, mic_q):
    mic_pq = mic_p - mic_q
    return mic_pq


def shd_add_noise(n, m):
    em32 = mc.EigenmikeEM32()
    estr = em32.returnAsStruct()
    ths = estr['thetas']
    phs = estr['phis']
    Ynm = np.random.rand(32, ) * 1j
    for ind in range(32):
        tq = ths[ind]
        pq = phs[ind]
        Ynm[ind] = sparg_sph_harm(m, n, pq, tq)
    return Ynm


def discorthonormality(N):
    em32 = mc.EigenmikeEM32()
    estr = em32.returnAsStruct()
    ths = estr['thetas']
    phs = estr['phis']
    CFnm = []
    for n in range(N + 1):
        for m in range(-n, n + 1):
            tmp = 0j
            for q in range(32):
                tmp += sparg_sph_harm(m, n, phs[q], ths[q]) * np.conj(sparg_sph_harm(m, n, phs[q], ths[q]))
            CFnm.append(1 / tmp)
    return np.array(CFnm)


def shd_all(channels, Nmax, k, a):
    Pnm = []
    for n in range(Nmax + 1):
        jn = sp.spherical_jn(n, k * a)
        jnp = sp.spherical_jn(n, k * a, derivative=True)
        yn = sp.spherical_yn(n, k * a)
        ynp = sp.spherical_yn(n, k * a, derivative=True)
        hn2 = jn + 1j * yn
        hn2p = jnp + 1j * ynp
        bnkra = jn - (jnp / hn2p) * hn2
        for m in range(-n, n + 1):
            pnm = shd_nm(channels, n, m) * ((-1) ** n) / (bnkra * 4 * np.pi * 1j ** n)
            Pnm.append(pnm)
    return Pnm


def Y_nm(n, m):
    em32 = mc.EigenmikeEM32()
    estr = em32.returnAsStruct()
    ths = estr['thetas']
    phs = estr['phis']
    Ynm = np.zeros(32) * 1j
    for ind in range(32):
        tq = ths[ind]
        pq = phs[ind]
        Ynm[ind] = sparg_sph_harm(m, n, pq, tq)
    return Ynm


def sparg_sph_harm(m, n, phi, theta):
    sph = (-1) ** m * np.sqrt((2 * n + 1) / (4 * np.pi) * sp.factorial(n - np.abs(m)) / sp.factorial(n + np.abs(m))) * \
          sp.lpmn(int(np.abs(m)), int(n), np.cos(theta))[0][-1][-1] * np.exp(1j * m * phi)
    # lpmn ve slicing yerine lpmv kullanabilirmişiz sp.lpmv(n, n, 0)  =  sp.lpmv(n, n, 0)[0][-1][-1], latter is giving a matrix of whole values up to order/degree n.
    return sph


def sparg_sph_harm_list(m, n, phi, theta):
    s = []
    for i in range(len(phi)):
        ph = phi[i]
        th = theta[i]
        sph = (-1) ** m * np.sqrt(
            (2 * n + 1) / (4 * np.pi) * sp.factorial(n - np.abs(m)) / sp.factorial(n + np.abs(m))) * \
              sp.lpmn(int(np.abs(m)), int(n), np.cos(th))[0][-1][-1] * np.exp(1j * m * ph)
        s.append(sph)
    return (np.array(s))


def L_dipole(n, a, k, rpq_p_sph):
    """
    Function to calculate coupled sphere (i.e. L12, L21)
    Utilisated to create L matrix elements
    :param n: Spherical harmonics order
    :param a: radius
    :param k: wave number
    :param rpq_p_sph: q pole to p pole distance (spherical coordinate)
    :return: L for two poles
    """
    Lmax = n + 2
    Nmax = n + 2
    sqr_size = (Lmax - 1) ** 2
    jhlp = []
    L = np.zeros((n * (Lmax - 1) ** 2, n * (Lmax - 1) ** 2),
                 dtype=complex)  # np.full((n*(Lmax - 1) ** 2, n*(Lmax - 1) ** 2), np.arange(1.0,19.0))
    jhlp_fin = np.zeros((n * (Lmax - 1) ** 2, n * (Lmax - 1) ** 2), dtype=complex)  # L = np.arange(324.).reshape(18,18)

    l = n
    hnp_arr = []
    for i in range(l + 1):
        jnp = sp.spherical_jn(i, k * a, derivative=True)
        ynp = sp.spherical_yn(i, k * a, derivative=True)
        hnp = jnp + 1j * ynp
        for ind in range((i) * 2 + 1):
            jhlp.append(jnp / hnp)
            hnp_arr.append(hnp)
    jhlp = np.array(jhlp)
    L = np.eye(sqr_size)

    s = (n + 1) ** 2

    jhlp_fin = mtlb.repmat(jhlp, s, 1)
    SR = recur.reexp_coef(k, Lmax, Nmax, rpq_p_sph)
    L = SR.copy() * jhlp_fin
    return L


def get_key(poles, val):
    for key, value in poles.items():
        if val[0] == value[0] and val[1] == value[1] and val[2] == value[2]:
            k = key
            return k


def L_multipole(ordd, a, k, mics):
    """
    :param deg: Spherical harmonic order
    :param a: radius of sphere = 0.042
    :param k: wave number
    :param poles: Coordinates of multipoles = mic locations
    :return: Reexpension coefficient (SR) multipoles
    """
    sqr_size = (ordd + 1) ** 2  # (Lmax - 1) ** 2
    key, mic_locs = zip(*mics.items())
    Lsize = max(key)
    L_matrix = np.eye(sqr_size * Lsize, dtype=complex)

    for row in key:
        for col in key:
            if row == col:
                L = np.eye(sqr_size)
            else:
                rq_p = mics.get(row)
                rp_p = mics.get(col)
                rpq_p = mic_sub(rq_p, rp_p)
                rpq_p_sph = cart2sph(rpq_p[0], rpq_p[1], rpq_p[2])
                L = L_dipole(ordd, a, k, rpq_p_sph)
            L_matrix[((row - 1) * sqr_size):((row) * sqr_size), ((col - 1) * sqr_size):((col) * sqr_size)] = L
    return L_matrix


def D_multipole(C, mics, n, k, a):
    """
    C to D
    """
    size = (n + 1) ** 2
    jhnp = []
    for i in range(n + 1):
        jnp = sp.spherical_jn(i, k * a, derivative=True)
        ynp = sp.spherical_yn(i, k * a, derivative=True)
        hnp = jnp + 1j * ynp
        for i in range((i) * 2 + 1):
            jhnp.append(jnp / hnp)
    jhnp = np.array(jhnp)
    jhnp_resized = np.resize(jhnp, size * len(mics))
    D_flat = C * -jhnp_resized
    return D_flat


def C_multipole(N, freq, s_sph, k, mics, flag, rot):
    rho = 1.225
    key, mic_locs = zip(*mics.items())
    size = (N + 1) ** 2
    c = 0
    t_size = size * len(mics)
    C_input = np.zeros(t_size) * 1j
    c = 0
    phase_list = np.zeros(t_size) * 1j
    for keys in itertools.product(mic_locs):
        q = keys[0]
        rq_p = q
        if flag == "same cin":
            src_inward_sph = cart2sph(-src[0], -src[1], -src[2])  # r,th,phi
            src_inward_cart = -src
        elif flag == "pw calibrated cin":
            mic_src = (q - src)
            src_inward_sph = cart2sph(mic_src[0], mic_src[1], mic_src[2])
            src_inward_cart = mic_src
        else:
            print("no such flag exists")

        k_vec = k * src_inward_cart
        phase = np.exp(-np.sum(k_vec * rq_p) * 1j)
        for n in range(N + 1):
            for m in range(-n, n + 1):
                Ynm_s = sparg_sph_harm(m, n, src_inward_sph[2], src_inward_sph[1]).round(10)
                t = c * size + (n + 1) ** 2 - (n - m)
                anm = np.conj(Ynm_s) * np.exp(-1j * rot * m)
                Cnm = anm * 4 * np.pi * (1j) ** n * phase / (1j * 2 * np.pi * freq * -rho)
                C_input[t - 1] = Cnm
                phase_list[t - 1] = phase
        c += 1
        C_flat = C_input
    return C_flat, phase_list


def A_multipole(L, D, n):
    """
    :param L: Reexpension coefficient matrix
    :param D:
    :param n:
    :return:
    """
    lu, piv = LA.lu_factor(L)
    A_nm = LA.lu_solve((lu, piv), D)  # Eq.39
    size = (n + 1) ** 2
    return A_nm[0:size], A_nm


def pressure_withA(n_low, a, k, Anm):
    """
    Calculates pressure from spherical harmonic coefficients
    :param n_low: Spherical harmonics order
    :param a: Radius of sphere
    :param k: wave number
    :param Anm: Spherical Harmonics Coefficient (Matrix form)
    :return: Pressure for each microphones
    """
    rho = 1.225  # Density of air
    c = 343  # Speed of sound
    em32 = mc.EigenmikeEM32()
    estr = em32.returnAsStruct()
    ths = estr['thetas']
    phs = estr['phis']
    mic32 = np.zeros(32) * 1j
    for ind in range(32):
        tq = ths[ind]
        pq = phs[ind]
        potential = 0
        for n in range(0, n_low + 1):
            jnp = sp.spherical_jn(n, k * a, derivative=True)
            for m in range(-n, n + 1):
                Ynm = sparg_sph_harm(m, n, pq, tq)
                t = (n + 1) ** 2 - (n - m) - 1
                potential += Anm[t] * Ynm / (jnp)  # sph_harm(m, n, pw, tw) #Gumerov, Eq. 18
        pressure = -potential * c * rho / (k * a ** 2)
        mic32[ind] = pressure
    return mic32


def pressure_withA_multipole(n_low, a, k, Anm_all, no_of_rsmas):
    """
    Calculates pressure from spherical harmonic coefficients
    :param n_low: Spherical harmonics order
    :param a: Radius of sphere
    :param k: wave number
    :param Anm: Spherical Harmonics Coefficient (Matrix form)
    :return: Pressure for each microphones
    """
    rho = 1.225  # Density of air
    c = 343  # Speed of sound
    em32 = mc.EigenmikeEM32()
    estr = em32.returnAsStruct()
    ths = estr['thetas']
    phs = estr['phis']
    pressure_all = []
    size = (n_low + 1) ** 2
    for arr in range(no_of_rsmas):
        Anm = Anm_all[arr * size:arr * size + size]
        mic32 = np.zeros(32) * 1j
        for ind in range(32):
            tq = ths[ind]
            pq = phs[ind]
            potential = 0
            for n in range(0, n_low + 1):
                jnp = sp.spherical_jn(n, k * a, derivative=True)
                for m in range(-n, n + 1):
                    Ynm = sparg_sph_harm(m, n, pq, tq)
                    t = (n + 1) ** 2 - (n - m) - 1
                    potential += Anm[t] * Ynm / (jnp)  # sph_harm(m, n, pw, tw) #Gumerov, Eq. 30
            pressure = -potential * c * rho / (k * (a ** 2))
            mic32[ind] = pressure
        pressure_all.append(mic32)
    return pressure_all


def shd_nm(channels, n, m):
    em32 = mc.EigenmikeEM32()
    estr = em32.returnAsStruct()
    wts = estr['weights']
    ths = estr['thetas']
    phs = estr['phis']
    pnm = np.zeros(np.shape(channels[0])) * 1j
    for ind in range(32):
        cq = channels[ind]
        wq = wts[ind]
        tq = ths[ind]
        pq = phs[ind]
        Ynm = sparg_sph_harm(m, n, pq, tq)  # Rafaely Ynm
        pnm += wq * cq * np.conj(Ynm)
    return pnm


def shd_all2(channels, Nmax, k, a):
    Pnm = []
    rho = 1.225
    c = 343
    for n in range(Nmax + 1):
        jnp = sp.spherical_jn(n, k * a, derivative=True)
        for m in range(-n, n + 1):
            pnm = shd_nm(channels, n, m) * jnp * k * a ** 2 / (-rho * c)
            Pnm.append(pnm)
    return np.array(Pnm) * discorthonormality(Nmax)


def pressure_to_Anm(presmulti_n, n_max, no_of_rsmas, k, a):
    Anm_scatter = []
    for arr in range(no_of_rsmas):
        pressure_temp = presmulti_n[arr]
        Anm_scat_temp = np.array(shd_all2(pressure_temp, n_max, k, a)).flatten()
        Anm_scatter.append(Anm_scat_temp)
    Anm_scatter = np.array(Anm_scatter).flatten()
    return Anm_scatter


def Anm_to_D(Anm, L):
    D = np.dot(L, Anm)
    return D


def D_to_Cin(D, mics, jhnp, n):
    size = (n + 1) ** 2
    jhnp_resized = np.resize(jhnp, size * len(mics))
    C_in_scat = D * (1 / -jhnp_resized)
    return C_in_scat


def C_tilde_to_Anm(C, f, rho, mics):
    w = 2 * np.pi * f
    block = int(len(C) / len(mics))
    n = int(np.sqrt(block) - 1)
    C_split = C.reshape(len(mics), block)
    anm = []
    for row in range(len(C_split)):
        C_single = C_split[row]
        for n in range(0, n + 1):
            for m in range(-n, n + 1):
                anm.append(C_single * -1j * w * rho / (4 * np.pi * ((1j) ** n)))
    anm = np.array([anm]).flatten()
    return (anm)


def pfield_sphsparg(f, k, mesh_row, N, C_in):
    """ extrapolated pressure at distance r with given anm's [as a list]"""
    mesh_sph, r = cart2sphr(mesh_row)
    th = mesh_sph[:, 0]
    ph = mesh_sph[:, 1]
    rho = 1.225
    pr = 0
    kr = k * r
    w = 2 * np.pi * f

    count = 0
    pr = 0
    for n in range(N + 1):
        for m in range(-n, n + 1):
            term = C_in[count] * (-1j * w * rho) * spherical_jn(n, kr, derivative=False) * sparg_sph_harm_list(m, n, ph,
                                                                                                               th)
            pr = pr + term
            count += 1
    return (pr)


def pfield_sphsparg_point(f, k, mesh_row, N, C_in):
    """ extrapolated pressure at distance r with given anm's [as a list]"""

    r, th, ph = cart2sph_single(mesh_row)
    rho = 1.225
    pr = 0
    kr = k * r
    w = 2 * np.pi * f

    count = 0
    pr = 0
    for n in range(N + 1):
        for m in range(-n, n + 1):
            term = C_in[count] * (-1j * w * rho) * spherical_jn(n, kr, derivative=False) * sparg_sph_harm(m, n, ph, th)
            pr = pr + term
            count += 1
    return (pr)


def plot_contour_DELAY(pressure, x, vsize):
    """ contour plot (for pressure or angular error) with a 2d meshgrid """
    fig, ax = plt.subplots(figsize=(7, 7))
    fig.gca().set_xlabel(r'$\tau_p\;[ms]$', fontsize=16)
    fig.gca().set_ylabel(r'$\tau_q\;[ms]$', fontsize=16)

    l = int(np.sqrt(len(pressure)))
    pressure_shaped = pressure.reshape(l, -1)
    pressure_real = pressure_shaped.real

    r_xx, r_yy = np.mgrid[-x:x:(vsize * 1j), -x:x:(vsize * 1j)]
    # t = plt.contour(r_xx,r_yy, pressure_real, cmap="jet")

    CS = ax.contour(r_xx * 10 ** 3, r_yy * 10 ** 3, pressure_real * 10 ** 3, cmap='binary')
    ax.contourf(r_xx * 10 ** 3, r_yy * 10 ** 3, pressure_real * 10 ** 3,
                cmap='Spectral')  # Wistia  afmhot gist_yarg autumn
    ax.clabel(CS, inline=True, fontsize=18)
    ax.set_aspect("equal")

    # ax.set_facecolor('xkcd:salmon')
    # p2 = ax.get_position().get_points().flatten()
    # ax_cbar1 = fig.add_axes([p2[0],p2[2], p2[2]-p2[0], 0.025])
    # plt.colorbar(t,cax=ax_cbar1 ,orientation="horizontal",ticklocation = 'top')
    return (ax)


def plot_scene(src, mics):
    key, mic_locs = zip(*mics.items())
    ad, bd, cd = list(zip(*mic_locs))
    ar, br, cr = list(zip(src))
    fig = plt.figure()
    ax50 = Axes3D(fig)
    ax50.scatter(ad, bd, cd, s=100)
    ax50.scatter(ar, br, cr, c="r", s=250)
    return


def rotvec(phis, N, Q):
    q = []
    if not len(phis) == Q:
        pad = Q - len(phis)
        phis = np.pad(phis, (0, pad), 'constant', constant_values=(0))
    for mic in range(Q):
        for n in range(N + 1):
            for m in range(-n, n + 1):
                q.append(np.exp(1j * m * phis[mic]))
    return np.array(q)


def rotatemat(MPmat, qrot):
    return qrot @ MPmat


def cin_roterror_iter(Anm_tilde, L, n):
    o = (n + 1) ** 2
    err = []
    p_rot = np.linspace(0, 2 * np.pi, 360)

    for rot in p_rot:
        Anm_tilde_rot = rotate_anm(Anm_tilde, rot, mics)
        D_tilde = Anm_to_D(Anm_tilde_rot, L)
        C_in_tilde = D_to_Cin(D_tilde, mics, jhnp, n)

        c1_tilde = C_in_tilde[0:o]
        c2_tilde = C_in_tilde[o:2 * o]
        err.append(np.linalg.norm(c1_tilde - c2_tilde))

    err_ = np.array(err)
    ind = err.index(min(err))
    print("order:", n, ", cin rotation iteration:")
    print("min hatanın açısı=", np.degrees(p_rot[ind]), "derece")
    print("max error:", np.max(err_))
    print("min error:", np.min(err_))
    fig = plt.figure()
    plt.plot(p_rot, err)
    plt.title("rotation iteration")
    return


def cin_delerror_3D(Anm_tilde, L, n, p_del, q_del):
    pX, pY = np.meshgrid(p_del, q_del)
    size_r = (n + 1) ** 2  # real shd block size
    index_list = mzeros_index(n)
    size = len(index_list)  # size of only m = 0 indexes
    err_rmse = []
    c = 0
    for prot in p_del:
        for qrot in q_del:
            c += 1
            delay_list = [prot, qrot, 0]
            Anm_tilde_del = delay_anm_3D(Anm_tilde, delay_list, mics, n, f)
            D_tilde = Anm_to_D(Anm_tilde_del, L)
            C_in_tilde = D_to_Cin(D_tilde, mics, jhnp, n)

            cr = []
            for t in range(len(mics)):
                for ind in index_list:
                    cr.append(C_in_tilde[ind + (t * size_r):ind + (t * size_r) + 1])

            C_in_tilde_reduced = np.array(cr).flatten()
            comb = combinations(range(1, len(mics) + 1), 2)
            err_tot = 0

            for i, j in list(comb):
                term = np.linalg.norm(
                    C_in_tilde_reduced[(i - 1) * size:i * size] - C_in_tilde_reduced[(j - 1) * size:j * size]) ** 2
                err_tot = err_tot + term

            err_tot = (np.sqrt(err_tot / len(mics)))
            err_rmse.append(err_tot)
    return (err_rmse)


def clamp(a, amax):
    if a > amax:
        return amax
    else:
        return a


def ordervec(fr, ra, Nmax):
    c = 341.
    fcnt = len(fr)
    kra = np.abs(2 * np.pi * fr / c * ra)
    orderlist = []
    for find in range(fcnt):
        krai = kra[find]
        orderlist.append(int(clamp(np.round(krai), Nmax)))
    return orderlist


def jhnp_func(n, k, a):
    jhnp = []
    for i in range(n + 1):
        jnp = sp.spherical_jn(i, k * a, derivative=True)
        ynp = sp.spherical_yn(i, k * a, derivative=True)
        hnp = jnp + 1j * ynp
        for i in range((i) * 2 + 1):
            jhnp.append(jnp / hnp)
    jhnp = np.array(jhnp)
    return (jhnp)


def rotate_anm(Anm_tilde, azi_rot, mics):
    Anm_tilde_diag = np.diag(Anm_tilde)
    qr = rotvec([azi_rot], n, len(mics))
    Anm_tilde_rot = rotatemat(Anm_tilde_diag, qr)
    return (Anm_tilde_rot)


def rotate_anm_3D(Anm_tilde, rot_list, mics):
    Anm_tilde_diag = np.diag(Anm_tilde)
    qr = rotvec(rot_list, n, len(mics))
    Anm_tilde_rot = rotatemat(Anm_tilde_diag, qr)
    return (Anm_tilde_rot)


def delay_anm_3D(Anm_tilde, delay_list, mics, order, f):
    if not len(delay_list) == len(mics):
        pad = len(mics) - len(delay_list)
        delay_list = np.pad(delay_list, (0, pad), 'constant', constant_values=(0))

    w = 2 * np.pi * f
    Anm_tilde_diag = np.diag(Anm_tilde)
    q = []
    for mic in range(len(mics)):
        for n in range(order + 1):
            for m in range(-n, n + 1):
                q.append(np.exp(1j * w * delay_list[mic]))
    Anm_tilde_delayed = Anm_tilde_diag @ q
    return (Anm_tilde_delayed)


def find_rot(Anm_tilde_rot, true_azi_rot, n):
    """ rotates anm iteratively and finds the minimum error between two cin's (two rsmas) """
    o = (n + 1) ** 2
    p_rot = np.linspace(0, 2 * np.pi, 360)
    err_r_r = []
    for rot in p_rot:
        Anm_tilde_r_r = rotate_anm(Anm_tilde_rot, rot, mics)
        D_tilde_r = Anm_to_D(Anm_tilde_r_r, L)
        C_in_tilde_r = D_to_Cin(D_tilde_r, mics, jhnp, n)
        c1_tilde_r_r = C_in_tilde_r[0:o]
        c2_tilde_r_r = C_in_tilde_r[o:2 * o]
        err_r_r.append(np.linalg.norm(c1_tilde_r_r - c2_tilde_r_r))

    err_ = np.array(err_r_r)
    ind_min = err_r_r.index(min(err_r_r))
    print("\nfinding rotation angle:")
    print("order=", n, "freq=", f, "Hz")
    print("angle need to be found=", np.degrees(true_azi_rot))
    print("min error:", np.min(err_), "at index:", ind_min, "angle [degree]:", np.degrees(p_rot[ind_min]), "\n")

    plt.figure()
    plt.plot(p_rot, err_r_r)
    plt.title("finding rotation")
    return


def find_rot_3(Anm_tilde_rot, true_azi_rot, n):
    """ rotates anm iteratively and finds the minimum errors between two cin's of 3 rsmas """
    o = (n + 1) ** 2
    p_rot = np.linspace(0, 2 * np.pi, 360)
    err_r_r12 = []
    err_r_r13 = []
    err_r_r23 = []
    for rot in p_rot:
        Anm_tilde_r_r = rotate_anm(Anm_tilde_rot, rot, mics)
        D_tilde_r = Anm_to_D(Anm_tilde_r_r, L)
        C_in_tilde_r = D_to_Cin(D_tilde_r, mics, jhnp, n)
        c1_tilde_r_r = C_in_tilde_r[0:o]
        c2_tilde_r_r = C_in_tilde_r[o:2 * o]
        c3_tilde_r_r = C_in_tilde_r[2 * o:3 * o]

        err_r_r12.append(np.linalg.norm(c1_tilde_r_r - c2_tilde_r_r))
        err_r_r13.append(np.linalg.norm(c1_tilde_r_r - c3_tilde_r_r))
        err_r_r23.append(np.linalg.norm(c2_tilde_r_r - c3_tilde_r_r))

    err12_ = np.array(err_r_r12)
    err13_ = np.array(err_r_r13)
    err23_ = np.array(err_r_r23)

    ind_min12 = err_r_r12.index(min(err_r_r12))
    ind_min13 = err_r_r13.index(min(err_r_r13))
    ind_min23 = err_r_r23.index(min(err_r_r23))

    # print("\nfinding rotation angle:")
    print("order=", n, "freq=", f, "Hz")
    print("angle need to be found=", np.degrees(true_azi_rot))
    print("min error for c12:", np.min(err12_), "at index:", ind_min12, "angle [degree]:", np.degrees(p_rot[ind_min12]))
    print("min error for c13:", np.min(err13_), "at index:", ind_min13, "angle [degree]:", np.degrees(p_rot[ind_min13]))
    print("min error for c12:", np.min(err23_), "at index:", ind_min23, "angle [degree]:", np.degrees(p_rot[ind_min23]),
          "\n")

    plt.figure()
    plt.plot(p_rot, err_r_r12)
    plt.plot(p_rot, err_r_r13)
    plt.plot(p_rot, err_r_r23)
    plt.legend(["|c1-c2|", "|c1-c3|", "|c2-c3|"])
    plt.title("finding rotation")
    return


def CDLADC_tilde(C_in, mics, n, k, a, jhnp, SNR):  # L'yi soldan R-1 ile çarpabiliriz
    """ main calculations of gumerovs + spargs algorithm
    input: c_in for plane wave
    output: c_in_tilde according to eigenmic configuration """
    key, _ = zip(*mics.items())
    no_of_rsmas = len(key)

    D = D_multipole(C_in, mics, n, k, a)
    L = L_multipole(n, a, k, mics)
    _, Anm_all = A_multipole(L, D, n)
    presmulti = pressure_withA_multipole(n, a, k, Anm_all, no_of_rsmas)
    presmulti_n = add_noise(presmulti, SNR, no_of_rsmas)
    Anm_tilde = pressure_to_Anm(presmulti_n, n, no_of_rsmas, k, a)
    D_tilde = Anm_to_D(Anm_tilde, L)
    C_in_tilde = D_to_Cin(D_tilde, mics, jhnp, n)
    return (C_in_tilde, Anm_tilde, L)


def Anmreal_cin_tilde(anm, mics, n, k, a, jhnp):
    L = L_multipole(n, a, k, mics)
    D_tilde = Anm_to_D(anm, L)
    C_in_tilde = D_to_Cin(D_tilde, mics, jhnp, n)

    return (C_in_tilde, L)


def ADC_tilde(Anm, L, jhnp, mics, n):
    """ input: anm_tilde output: c_tilde """
    D_tilde = Anm_to_D(Anm, L)
    C_in_tilde = D_to_Cin(D_tilde, mics, jhnp, n)
    return (C_in_tilde)


def add_noise(pressure, SNR, no_of_poles):
    """
    :param mic32: pressure at each mic
    :param SNR: Signal-to-Noise ratio (dB)
    :return:
    """
    for i in range(no_of_poles):
        pres_temp = pressure[i]
        noise = np.random.rand(32, )
        noise_r = np.random.randn(32)
        noise_i = np.random.randn(32)
        noise_r = noise_r - np.mean(noise_r)
        noise_i = noise_i - np.mean(noise_i)
        noise = noise_r + noise_i * 1j
        mic_norm = np.linalg.norm(pres_temp, axis=0)
        noisy_pres = 0
        noise_norm = np.linalg.norm(noise, axis=0)
        coef = mic_norm / noise_norm
        SNR_linear = 10 ** (-SNR / 20)
        noise = noise * coef * SNR_linear
        pressure[i] = pres_temp + noise
    noisy_pres = pressure
    return noisy_pres


def cin_scaling(mics, src, N):
    key, mic_locs = zip(*mics.items())
    mic_src_list = []
    size = (N + 1) ** 2
    for keys in itertools.product(mic_locs):
        mic = keys[0]
        mic_src = mic - src
        mic_src_list.append(-mic_src)

    mic_src_sph = cart2sphr_sparg(np.array(mic_src_list))
    ynmz = []
    for n in range(N + 1):
        for m in range(-n, n + 1):
            Ynm_s = sparg_sph_harm_list(m, n, mic_src_sph[2], mic_src_sph[1])
            # Ynm_s = sph_harm(m, n, mic_src_sph[1], mic_src_sph[2])
            # Ynm_s = sph_harm(m, n, mic_src_sph[:,1], mic_src_sph[:,0])
            ynmz.append(np.conj(Ynm_s))
    ynm = np.array(ynmz).reshape(size, len(mics))
    return (ynm, mic_src_sph)


def total_cin_err(rot_list, Anm_tilde_rotated, L, n, mics):
    ynm_scale, _ = cin_scaling(mics, src, n)
    rot_list = np.array(rot_list)
    size = (n + 1) ** 2
    Anm_tilde_rot_rot = rotate_anm_3D(Anm_tilde_rotated, rot_list, mics)
    C_in_tilde = ADC_tilde(Anm_tilde_rot_rot, L, jhnp, mics, n)

    comb = combinations(range(1, len(mics) + 1), 2)
    err_sum = 0
    for i, j in list(comb):
        term = np.linalg.norm(C_in_tilde[(i - 1) * size:i * size] * ynm_scale[:, (j - 1)].round(10) - C_in_tilde[(j - 1) * size:j * size] * ynm_scale[:,(i - 1)].round(
            10)) ** 2
        err_sum = err_sum + term
    err_rmse = (np.sqrt(err_sum / len(mics)))
    return (err_rmse)


def mzeros_index(order):
    index = 0
    ind = []
    for n in range(order + 1):
        for m in range(-n, n + 1):
            if m == 0:
                ind.append(index)
            index += 1
    return (ind)


def total_cin_delayerr(delay_list, Anm_tilde_delayed, L, n, mics, f):
    delay_list = np.array(delay_list)
    size_r = (n + 1) ** 2  # real shd block size

    index_list = mzeros_index(n)
    size = len(index_list)  # size of only m = 0 indexes

    Anm_tilde_del_del = delay_anm_3D(Anm_tilde_delayed, delay_list, mics, n, f)
    C_in_tilde = ADC_tilde(Anm_tilde_del_del, L, jhnp, mics, n)
    cr = []
    for t in range(len(mics)):
        for ind in index_list:
            cr.append(C_in_tilde[ind + (t * size_r):ind + (t * size_r) + 1])

    C_in_tilde_reduced = np.array(cr).flatten()
    comb = combinations(range(1, len(mics) + 1), 2)
    err_sum = 0

    for i, j in list(comb):
        term = np.linalg.norm(
            C_in_tilde_reduced[(i - 1) * size:i * size] - C_in_tilde_reduced[(j - 1) * size:j * size]) ** 2
        err_sum = err_sum + term

    err_rmse = (np.sqrt(err_sum / len(mics)))
    return (err_rmse)


def total_cin_delayerr_fiter(delay_list, Anm_tilde_delayed, L, n, mics, nf, fmin, fmax):
    farr = np.linspace(fmin, fmax, nf)

    err_rsme = 0
    for f in farr:
        delay_list = np.array(delay_list)
        size_r = (n + 1) ** 2  # real shd block size

        index_list = mzeros_index(n)
        size = len(index_list)  # size of only m = 0 indexes

        Anm_tilde_del_del = delay_anm_3D(Anm_tilde_delayed, delay_list, mics, n, f)
        C_in_tilde = ADC_tilde(Anm_tilde_del_del, L, jhnp, mics, n)
        cr = []
        for t in range(len(mics)):
            for ind in index_list:
                cr.append(C_in_tilde[ind + (t * size_r):ind + (t * size_r) + 1])

        C_in_tilde_reduced = np.array(cr).flatten()
        comb = combinations(range(1, len(mics) + 1), 2)
        err_sum = 0

        for i, j in list(comb):
            term = np.linalg.norm(
                C_in_tilde_reduced[(i - 1) * size:i * size] - C_in_tilde_reduced[(j - 1) * size:j * size]) ** 2
            err_sum = err_sum + term

        err_rsme = err_rsme + (np.sqrt(err_sum / len(mics)))
    return (err_rsme)

def total_cin_delayerr_fiter_real(delay_list, Anm_tilde_delayed, Larr, n, mics, farr, a):
    rw = len(farr)
    err_rsme = 0

    for ind in range(rw):
        f = farr[ind]
        k = 2 * pi * f / c
        jhnp = jhnp_func(n, k, a)
        delay_list = np.array(delay_list)
        size_r = (n + 1) ** 2  # real shd block size

        index_list = mzeros_index(n)
        size = len(index_list)  # size of only m = 0 indexes

        Anm_tilde_del_del = delay_anm_3D(Anm_tilde_delayed[:, ind], delay_list, mics, n, f)
        C_in_tilde = ADC_tilde(Anm_tilde_del_del, Larr[ind], jhnp, mics, n)
        cr = []
        for t in range(len(mics)):
            for ind in index_list:
                cr.append(C_in_tilde[ind + (t * size_r):ind + (t * size_r) + 1])

        C_in_tilde_reduced = np.array(cr).flatten()
        comb = combinations(range(1, len(mics) + 1), 2)
        err_sum = 0

        for i, j in list(comb):
            term = np.linalg.norm(
                C_in_tilde_reduced[(i - 1) * size:i * size] - C_in_tilde_reduced[(j - 1) * size:j * size]) ** 2
            # term /= np.linalg.norm(
            #    C_in_tilde_reduced[(i - 1) * size:i * size] + C_in_tilde_reduced[(j - 1) * size:j * size]) ** 2
            err_sum = err_sum + term

        err_rsme = err_rsme + (np.sqrt(err_sum / len(mics)))
    return (err_rsme)

def toa(mics, src, fs, c=341.):
    toa_list = []
    key, mic_locs = zip(*mics.items())
    for i in range(len(mic_locs)):
        toa_list.append(int(np.linalg.norm(src - mic_locs[i]) * fs / c))
    # print("toa_sample=", toa_list)
    return (toa_list)

def rawread(wave_file_dir):
    rate, raw = wavfile.read(wave_file_dir, mmap=False)
    row, col = np.shape(raw)
    numchan = col
    numlen = row
    # print(row)
    # print(col)
    return numlen, numchan, rate, raw

def raw2ambix(ref_data):
    # rate1, raw = wavfile.read(filepos, mmap=False)
    # irdir = '/Users/orhunsparg/PycharmProjects/somp-interpolation/data/A2B-Zylia-3E-Jul2020.wav'
    irdir = './data/A2B-Zylia-3E-Jul2020.wav'

    raw = ref_data
    rate2, eir = wavfile.read(irdir, mmap=False)
    # assert rate1 == rate2 # Make sure that the sampling rates are the same
    rowd, cold = np.shape(raw)  # Data
    rowi, coli = np.shape(eir)  # Impulse (FARINA FILTERS)
    irsize = int(rowi / (cold))  # Last channel is timecode, we will not process it here
    ambix = np.zeros((rowd + irsize - 1, coli), dtype=float)
    # print("row",rowd)
    # print("col",cold)
    for ind in range(coli):
        # ir = eir[:,ind]
        ir = eir[:, ind] / (2. ** 32)  # Raw recordings are 32-BIT DO NOT DELETE
        for jnd in range(cold):
            sig = raw[:, jnd]/ (2.**32)
            ire = ir[jnd * irsize: (jnd + 1) * irsize]
            eqsig = fftconvolve(sig, ire, mode='full')
            ambix[:, ind] += eqsig
    return ambix

def raw2ambix_single(ref_data):
    # rate1, raw = wavfile.read(filepos, mmap=False)
    # irdir = '/Users/orhunsparg/PycharmProjects/somp-interpolation/data/A2B-Zylia-3E-Jul2020.wav'
    irdir = './data/A2B-Zylia-3E-Jul2020.wav'

    raw = ref_data
    rate2, eir = wavfile.read(irdir, mmap=False)
    # assert rate1 == rate2 # Make sure that the sampling rates are the same
    rowd, cold = np.shape(raw)  # Data
    rowi, coli = np.shape(eir)  # Impulse (FARINA FILTERS)
    irsize = int(rowi / (cold))  # Last channel is timecode, we will not process it here
    ambix = np.zeros((rowd + irsize - 1, coli), dtype=float)
    for ind in range(coli):
        # ir = eir[:,ind]
        ir = eir[:, ind] / (2. ** 32)  # Raw recordings are 32-BIT DO NOT DELETE
        for jnd in range(cold):
            sig = raw[:, jnd]
            ire = ir[jnd * irsize: (jnd + 1) * irsize]
            eqsig = fftconvolve(sig, ire, mode='full')
            ambix[:, ind] += eqsig
    return ambix

def fdambix(ambixchans, nfft):
    # Ambix to Frequency domain translation
    print(ambixchans)
    rw, cl = np.shape(ambixchans)
    print("fdambix")
    print(rw)
    print(cl)
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


def listoflists(array):
    rw, cl = np.shape(array)

    samples = []
    ls = list(np.transpose(array))
    for ch in range(cl):
        samples.append(list(ls[ch]))
    return samples


def nm2acn(n, m):
    # Convert (n,m) indexing to ACN indexing
    return n ** 2 + n + m


def ambix2sh(ambixchans):
    rw, cl = np.shape(ambixchans)
    # print("ambix2sh")
    # print(rw)
    # print(cl)
    shchannels = np.zeros((rw, cl), dtype=complex)
    N = int(np.sqrt(cl) - 1)

    for n in range(N + 1):
        ngain = np.sqrt(2 * n + 1)
        sqrt2 = np.sqrt(2)
        for m in range(-n, n + 1):
            chanind1 = nm2acn(n, m)
            chanind2 = nm2acn(n, -m)
            if m < 0:
                shchannels[:, chanind1] = ngain / sqrt2 * (ambixchans[:, chanind2] + 1j * ambixchans[:, chanind1])
            elif m > 0:
                shchannels[:, chanind1] = (-1) ** m * ngain / sqrt2 * (
                            ambixchans[:, chanind1] - 1j * ambixchans[:, chanind2])
                # shchannels[:, chanind1] = ngain / sqrt2 * (ambixchans[:, chanind1] - 1j * ambixchans[:, chanind2])
            else:  # when m=0
                shchannels[:, chanind1] = ngain * ambixchans[:, chanind1]

    return shchannels


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


def total_cin_roterr_fiter_real(rot_list, anm_time_aligned, Larr, n, mics, farr, a):
    # multiple frequency optimisation for rotation (real signal)
    rw = len(farr)
    err_rsme = 0
    ynm_scale, _ = cin_scaling(mics, src, n)
    size = (n + 1) ** 2
    for ind in range(rw):
        f = farr[ind]
        # print("frequency: ", f)
        k = 2 * pi * f / c
        jhnp = jhnp_func(n, k, a)

        rot_list = np.array(rot_list)
        Anm_tilde_rot_rot = rotate_anm_3D(anm_time_aligned[:, ind], rot_list, mics)
        C_in_tilde = ADC_tilde(Anm_tilde_rot_rot, Larr[ind], jhnp, mics, n)
        # C_in_tilde = Anm_tilde_rot_rot
        comb = combinations(range(1, len(mics) + 1), 2)
        err_sum = 0
        for i, j in list(comb):
            term = np.linalg.norm(C_in_tilde[(i - 1) * size:i * size] * ynm_scale[:, (j - 1)].round(10) -
                                  C_in_tilde[(j - 1) * size:j * size] * ynm_scale[:, (i - 1)].round(10)) ** 2
            err_sum = err_sum + term
        err_rsme = err_rsme + (np.sqrt(err_sum / len(mics)))
    return (err_rsme)


def align(input, toa, offset=0):
    assert toa > -offset
    print(input.shape)
    out = input[int(toa + offset):, :]
    return out


def wav2shd_nmax(subdir, irdir, nmax):
    shd_list = []
    raw_list = []
    amb_list = []
    dr = os.path.join("./data", subdir)
    if FLAG == "orhun":
        dr = subdir
        print("flag==orhun, directory:",dr)
    rate = 0
    for filedir in glob.glob(dr):
        print(filedir)
        rate, abx, raw = amb.raw2ambix(filedir, irdir)
        raw_list.append(raw)
        amb_list.append(np.transpose(abx))
        fd = amb.fdambix(abx, 1024)
        fda = amb.fdlist2array(fd)
        fdsh = amb.ambix2sh(fda)
        shd_list.append(fdsh[:, :(nmax + 1) ** 2])
    return rate, shd_list, raw_list, amb_list

def wav2shd_fromarray(rawlist_orj, nmax, nfft):
    #assert nfft > len(rawlist_orj[0,:,0].flatten()) + 4096
    #raw19ch = rawlist_orj[0]
    shd_list = []
    amb_list = []

    for ind in range(len(rawlist_orj)):
        #print("SHAPE rawlist_orj[ind]", np.asarray(rawlist_orj[ind]).shape)
        abx = raw2ambix(rawlist_orj[ind,:,:])
        amb_list.append(np.transpose(abx))
        fd = amb.fdambix(abx, nfft)
        fda = amb.fdlist2array(fd)
        fdsh = amb.ambix2sh(fda)
        shd_list.append(fdsh[:, :(nmax + 1) ** 2])

    return shd_list, amb_list

def wav2shd_nmax_pre(subdir, irdir, nmax, mics, src, fs):
    shd_list = []
    offset = 0#-100
    raw_list = []
    toa_list = toa(mics, src, fs, c=341.)
    ind = 0
    os.chdir(subdir)
    path = os.getcwd()
    #dr = os.path.join(sbudir, subdir)
    rate = 0
    print(path)
    files = glob.glob("*.wav")
    print(sorted(files))
    for filedir in sorted(files):
        print(filedir)
        os.path.join(subdir, filedir)
        rate, abx, raw = amb.raw2ambix(filedir, irdir)
        #rate, abx= amb.raw2ambix(filedir, irdir)

        rawa = align(raw, toa_list[ind], offset)
        abxa = align(abx, toa_list[ind], offset)
        raw_list.append(np.transpose(rawa))
        ind += 1

        fd = amb.fdambix(abxa, 1024)
        fda = amb.fdlist2array(fd)
        fdsh = amb.ambix2sh(fda)
        shd_list.append(fdsh[:, :(nmax + 1) ** 2])
    return rate, shd_list, raw_list

def align_toa(raws, mics, src, fs):
    toa_list = toa(mics, src, fs, c=341.)
    print("toa delays:", toa_list)
    toa_aligned_wavs = shift_list(raws, toa_list)
    return(toa_aligned_wavs)

def align_toa_multi(raws, mics, src, fs):
    toa_list = toa(mics, src, fs, c=341.)
    print("toa delays:", toa_list)
    toa_aligned_wavs = shift_list_multi(raws, toa_list)
    return(toa_aligned_wavs)

def stackshd(shd_list, f_ind, n0):
    # stack mic shd's
    b = np.zeros((len(f_ind), (n0 + 1) ** 2))
    print("f_ind", f_ind)
    #anm = np.transpose((b[:, (n0 + 1) ** 2:]))
    for mic in shd_list:
        a = mic[f_ind, :]
        b = np.append(b, a, axis=1)

    anm = np.transpose(b[:, (n0 + 1) ** 2:])
    print(anm.shape)
    print("")
    return anm

def savewav(fileno, subdir, signal, rate):
    wav_name = 'shifted_%s_19ch.wav' % (fileno)
    wav_dir = os.path.join(subdir, wav_name)
    write(wav_dir, rate, signal)
    return None

def saveshifted(signal, fs, micNo, dir):
    os.chdir(dir)
    #ir = ir - ir[ind, :][0]
    filename = "shifted_" + str(micNo) + ".wav"
    wavfile.write(filename, fs, signal)  # .astype(np.float32))
    return None

def shift_list(rawlist, samplelist):
    # raw_list =  7x512 (channel no x signal length)
    shifted_list = []
    for ind in range(len(samplelist)):
        shifted_list.append(shift(rawlist[ind], samplelist[ind]))
    return(shifted_list)


def shift_list_multi(rawlist_arr, samplelist):
    # raw_list =  7x512 (channel no x signal length)
    shifted_list = []
    size = np.max(samplelist)
    for ind in range(len(samplelist)):
        shifted_list.append(shift_multi(rawlist_arr[ind, :, :], samplelist[ind], size))
    return(shifted_list)

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

def shift(raw0, shift_sample):
    if shift_sample < 0:
        assert "error"
    else:
        # print("shifted %d" %shift_sample )
        pre = np.pad(raw0, (shift_sample, 0), constant_values=0)
    return(pre)

def json2dict(filename):
    f = open(filename)
    return json.load(f)

def json2miclist_z(filename):
    f = open(filename)
    jdict = json.load(f)
    mic_dict = {}
    mic_dict_z = {}
    mic_list = []
    for key, value in jdict.items():
        if key[0:3]=="Pos":
            mic_dict[key] = value
            mic_list.append(value)
    mic_list_z = np.c_[mic_list, np.zeros(len(mic_dict))]
    for i in range(len(mic_list_z)):
        mic_dict_z[i] = mic_list_z[i]
    return mic_list_z, mic_dict_z

def list2dict(ls):
    mic_dict_z = {}
    for i in range(len(ls)):
        mic_dict_z[i + 1] = ls[i]
    return mic_dict_z

if __name__ == '__main__':

    # FLAG = "orhun"
    # FLAG = "ege"
    FLAG = "ege_imac"
    start_time = datetime.now()

    """ Ambixutil: wav-ambix-shd conversion """

    if FLAG == "ege":
        irdir = './data/A2B-Zylia-3E-Jul2020.wav'
        subdir = r"smallhexa_lab\30deg_IR\*"

    elif FLAG == "ege_imac":
        irdir = './data/A2B-Zylia-3E-Jul2020.wav'
        subdir = "smallhexa_lab/30deg_IR/*"

    elif FLAG == "orhun":
        irdir = '/Users/orhunsparg/PycharmProjects/somp-interpolation/data/A2B-Zylia-3E-Jul2020.wav'
        subdir = './data/multiple_trial_not_important/30degsmall/*'
        #subdir = './data/multiple_trial_not_important/largehexaref/*'

    # subdir = r"bighexa_lab\reference_IR\*"
    # subdir = r"bighexa_lab\reference_no_reflections\*"

    fs = 48000
    pi = np.pi
    n = 3  # Spherical Harmonic Order
    order = n  # Spherical Harmonic Order for L matrix
    a = 5.5e-2  # Radius of sphere  (42e-3 for eigenmic)

    c = 343  # Speed of sound

    # MDS LOCAL CODE LOCATIONs

    """   
    m1 = np.array([0.24713039, - 0.12799929, 8.85000000e-01])  # 1
    m2 = np.array([0.12284332, 0.06532657, 0.885])  # 2
    m3 = np.array([0.14605493, - 0.3343197, 0.885])  # 3
    m4 = np.array([0.01751808, - 0.13803243, 0.885])  # 4
    m5 = np.array([0.35732658, 0.07359733, 0.885])  # 5
    m6 = np.array([0.3769871, - 0.32379636, 0.885])  # 6
    m7 = np.array([0.48197453, - 0.12108931, 0.885])  # 7
    src_orh = np.array([-1.74983493, 0.90631319, 1.3])

    mics_orh = {1: m1, 2: m2, 3: m3, 4: m4, 5: m5, 6: m6, 7: m7}
    """

    # Load json-mic locs

    if FLAG == "ege":
        mic_list_z, mic_dict_z = json2miclist_z(".\data\smallhexalab.json")
        file = open("rotated_mic_list_z", "rb")
        rotated_mic_list_z = pickle.load(file)
        file.close()
    elif FLAG == "ege_imac":
        mic_list_z, mic_dict_z = json2miclist_z("./data/smallhexalab.json")
        file = open("rotated_mic_list_z", "rb")
        rotated_mic_list_z = pickle.load(file)
        file.close()
    elif FLAG == "orhun":
        mic_list_z, mic_dict_z = json2miclist_z("./data/smallhexalab.json")
    """
    # z axis of seven mics set to 0
    mic_list_z[0:7, 2] = 0.885
    # z axis of the source set to 1.3
    mic_list_z[7, 2] = 1.3
    # mic_list_z[7, 2] = 0.885

    mic_dict_z = list2dict(mic_list_z[0:7, :])
    # mic_dict_z = list2dict(mic_list_z[0:2, :])
    mics = mic_dict_z

    src = mic_list_z[7]
    rsrc_sph = cart2sph(src[0], src[1], src[2])  # source coordinates in spherical coors.
    """

    mic_dict_z = list2dict(rotated_mic_list_z[0:7, :])
    src = rotated_mic_list_z[7]
    rsrc_sph = cart2sph(src[0], src[1], src[2])  # source coordinates in spherical coors.

    mics = {1: mic_list_z[4], 2: mic_list_z[3], 3: mic_list_z[2], 4: mic_list_z[5], 5: mic_list_z[0], 6: mic_list_z[1], 7: mic_list_z[6]}
    # mics = {1: mic_list_z[0], 2: mic_list_z[1], 3: mic_list_z[2], 4: mic_list_z[3], 5: mic_list_z[4], 6: mic_list_z[5],
    #         7: mic_list_z[6]}

    # mics = {1: mic_list_z[2], 2: mic_list_z[3], 3: mic_list_z[5]}
    # plt.figure()
    # plot_scene(src, mics)
    # plt.xlim(-1, 1)
    # plt.ylim(-1, 1)
    # plt.show()

    # plt.figure()
    #plot_scene(src_orh, mics_orh)
    #plt.xlim(-1, 1)
    #plt.ylim(-1, 1)
    # plt.zlim(-1, 1)

    # rate, shd_list = wav2shd(subdir, irdir)
    rate, _, rawlist_orj, ambix_list = wav2shd_nmax(subdir, irdir, nmax=n)  # for different n's
    # rate, shd_list, wav_list = wav2shd_nmax_pre(subd, irdir, n, mics, src,fs)  # for different n's, geo delay prealigned

    delay_list = np.zeros(len(mics))
    for i in range(len(mics)-1):
        delay_list[i+1] = int(clb.corralign(ambix_list[0][0, :], ambix_list[i+1][0, :]))
        # print("i, i+1 ", 1, i+2)
    rawlist_arr = np.array(rawlist_orj)
    samplelist_neworg = np.array(delay_list) - np.min(delay_list)
    samplelist_neworg = samplelist_neworg.astype(int)  # crosscorr_del
    # shifted_raws_ZZ = shift_list(rawlist_arr, samplelist_neworg)
    shifted_raws_multichan = shift_list_multich(rawlist_arr, samplelist_neworg)
    shifted_arr_multi = np.array(shifted_raws_multichan)   # all aligned as same
    print("rawlist_arr_multishift", shifted_arr_multi.shape)
    print("crosscorrelation delay list:", delay_list)
    print("positive (new origin) delay_list:", samplelist_neworg)

    #toa_aligned_wavs = align_toa(shifted_raws, mics, src, fs)
    toa_aligned_multi = align_toa_multi(shifted_arr_multi, mics, src, fs)   # toa added to perfect alignment
    toa_aligned_multi_arr = np.array(toa_aligned_multi)

    plt.figure()
    plt.title("raws")
    for i in range(len(mics)):
        plt.plot(rawlist_arr[i,:,4])
    plt.show()

    plt.figure()
    plt.title("crosscorr.")
    for i in range(len(mics)):
        plt.plot(shifted_arr_multi[i,:,4])
    plt.show()

    plt.figure()
    plt.title("toa_aligned")
    for i in range(len(mics)):
        plt.plot(toa_aligned_multi_arr[i,:,4])
    plt.show()
    end_time = datetime.now()


    print('Part 1: Crosscorrelation Duration'.format(end_time - start_time))

    " ROTATION OPTIMISATION "

    #f_list = np.linspace(0, 48000 / 2, int(len(rawlist[0])/2))
    fft_block_size = 16384
    # fft_block_size = 1024
    freqs = fft.rfftfreq(fft_block_size, d=1 / 48000.0)
    # fr = np.array([1000,2000,3000])
    #n_list = ordervec(f_list, 5.5e-2, 3)

    fr_ind = freqs[1]-freqs[0] #fs / 2 / fft_block_size
    fr_ind = fs / 2 / fft_block_size
    print("freq step size=", fr_ind)
    # kra<1,2,3

    if n == 0:
        flow = 1
        fhigh = 652 / fr_ind
    elif n == 1:
        flow = 652 / fr_ind
        fhigh = 1301 / fr_ind
    elif n == 2:
        flow = 1301 / fr_ind
        fhigh = 2607 / fr_ind
    elif n == 3:
        # flow = 2607 / fr_ind
        flow = 1301 / fr_ind
        # fhigh = 4000 # upper_i
        fhigh = 2607  # upper_i
        # nd=300 since >20kHz was noisy

    # flow = 55
    # fhigh = 300
    np.random.seed(5)
    f_ind = np.random.randint(int(flow), int(fhigh), size=5)  # random seed = 15 ve flow= 55'da çalışmıyor???
    # print("flow", flow)
    # print("fhigh", fhigh)

    # f_ind = np.array([3593, 2334])
    # f_ind = np.array([31, 51])
    fr = freqs[np.unique(f_ind)]
    fr = freqs[f_ind]
    print("f_ind", f_ind)
    print("freqs:", fr)

    #dellist = []
    #er_del_square = np.zeros((90, 90))
    #size = (n + 1) ** 2

    print("\norder=", n)
    #del_err = dict()
    #del_error_list = []

    start_time = datetime.now()
    Larr = []
    for ind in range(len(fr)):
        k = 2 * pi * fr[ind] / c
        Larr.append(L_multipole(n, a, k, mics))
    end_time = datetime.now()
    print('Part 2: L Matrix Duration: {}'.format(end_time - start_time))

    # stack mic shd's 
    #flag = "toa off"
    #flag = "toa on"
    
    shd_list_toa, _ = wav2shd_fromarray(toa_aligned_multi_arr, nmax=n, nfft=16384)
    anm_time_aligned = stackshd(shd_list_toa, f_ind, n0=n)
    #anm_time_aligned = anm_time_aligned[0:(len(mics)*((n+1)**2)),:]

    print("anm_time_aligned shape:", anm_time_aligned.shape)
    #print(anm_time_aligned)
    # dtot = np.hstack((d1,d2,d3,d4,d5,d6,d7))
    # anm_time_aligned = np.transpose(dtot)

    rot_bnd = np.pi/3
    bnds = tuple([(-rot_bnd, rot_bnd)] * (len(mics)))
    # bnds = tuple([(-rot_bnd, rot_bnd)] * (len(mics)-1))

    # SINGLE FREQ ROTATION OPTIMISATION 
    # f_ind = 15
    # f = fr[f_ind]
    # rot list = number of mics = np.zeros((len(mics))
    # res = minimize(total_cin_err_real, np.zeros((len(mics))), (anm_time_aligned, Larr, n, mics, f, f_ind, a), method='Nelder-Mead')
    # print("minimum with only nelder-mead:", np.degrees(res.x))
    # minimizer_kwargs = {"method": "Nelder-Mead", "args": (anm_time_aligned, Larr, n, mics, f, f_ind, a), "bounds": bnds, "jac": False}
    # ret = basinhopping(total_cin_err_real, np.zeros(len(mics)), minimizer_kwargs=minimizer_kwargs, niter=50)
    # print("global minimum with basinhopping: x = ", np.degrees(ret.x), "\nf(x) = %.4f" % (ret.fun))

    # MULTIPLE FREQ ROTATION OPTIMISATION 

    start_time = datetime.now()
    print("optimisation has started:")

    #minimizer_kwargs = {"method": "Nelder-Mead", "args": (anm_time_aligned, Larr, n, mics, fr, a), "bounds": bnds,
    #                    "jac": False}
    #ret = basinhopping(total_cin_roterr_fiter_real, np.zeros(len(mics)), minimizer_kwargs=minimizer_kwargs, niter=50)
    #print("global minimum with basinhopping: x = ", np.degrees(ret.x), "\nf(x) = %.4f" % (ret.fun))

    params = (anm_time_aligned, Larr, n, mics, fr, a)
    res = differential_evolution(total_cin_roterr_fiter_real, bnds, args=params)
    print("global minimum with differential_evolution: x = ", np.degrees(res.x), "\nsuccess:", res.success, "\nres.fun:", res.fun)
    end_time = datetime.now()
    print('Part 3: Optimisation Duration: {}'.format(end_time - start_time))

    # x,f,y,z = brute(total_cin_roterr_fiter_real, bnds, args=params, full_output=True)
    # print("minimum with brute force: x = ", x)  #



