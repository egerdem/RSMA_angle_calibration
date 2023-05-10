from scipy.io import wavfile
import Microphone as mc
from scipy.signal import fftconvolve
import scipy.fft as fft
import scipy.special as sp
from scipy.special import spherical_jn, sph_harm
import numpy as np
import recursion_r7 as recur
from numpy import matlib as mtlb
from scipy import linalg as LA
from scipy.optimize import minimize
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import combinations
import pickle
from timeit import default_timer as timer
from scipy.optimize import basinhopping, brute, differential_evolution
#import seaborn as sns
import matplotlib
import ambixutil as amb
import json
import os
import glob
#import pandas as pd
#import seaborn as sns

# plt.close('all')
plt.close()

matplotlib.rc('xtick', labelsize=18)
matplotlib.rc('ytick', labelsize=18)

""" date: 24 Mart 22
    author: ege
    previously: multi_r13.py : both delay and rot. calib. with real signals, input is pickled ambix
    now: multi_r14.. : input is wav. ambix conversion is added.
    
    """

def cart2sphr(xyz): # Ege'
    """ converting a multiple row vector matrix from cartesian to spherical coordinates """
    c = np.zeros((xyz.shape[0],2))
    r = []
    tt = []
    for i in range(xyz.shape[0]):

        x       = xyz[i][0]
        y       = xyz[i][1]
        z       = xyz[i][2]
        
        rad       =  np.sqrt(x*x + y*y + z*z)
        tt.append(rad)
        theta   =  np.arccos(z/rad)
        phi     =  np.arctan2(y,x)
        c[i][0] = theta
        c[i][1] = phi    
        r.append(rad)   
        
    return [c, np.array(r)]

def cart2sph_single(xyz): #(ege)
    """ converting a single row vector from cartesian to spherical coordinates """
    #takes list xyz (single coord)
    x       = xyz[0]
    y       = xyz[1]
    z       = xyz[2]
    
    r       =  np.sqrt(x*x + y*y + z*z)
    theta   =  np.arccos(z/r)
    phi     =  np.arctan2(y,x)
    return(r, theta, phi)

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
    c = np.zeros((xyz.shape[0],2))
    r = []
    tt = []
    for i in range(xyz.shape[0]):
        
        
        x       = xyz[i][0]
        y       = xyz[i][1]
        z       = xyz[i][2]
        
        rad       =  np.sqrt(x*x + y*y + z*z)
        tt.append(rad)
        theta   =  np.arccos(z/rad)
        phi     =  np.arctan2(y,x)
        c[i][0] = theta
        c[i][1] = phi    
        r.append(rad)   
    return(np.array(r), c[:,0], c[:,1])
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

def shd_add_noise(n,m):
    em32 = mc.EigenmikeEM32()
    estr = em32.returnAsStruct()
    ths = estr['thetas']
    phs = estr['phis']
    Ynm = np.random.rand(32,)* 1j
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
    for n in range(N+1):
        for m in range(-n, n+1):
            tmp = 0j
            for q in range(32):
                tmp += sparg_sph_harm(m, n, phs[q], ths[q]) * np.conj(sparg_sph_harm(m, n, phs[q], ths[q]))
            CFnm.append(1/tmp)
    return np.array(CFnm)

def shd_all(channels, Nmax, k, a):
    Pnm = []
    for n in range(Nmax+1):
        jn = sp.spherical_jn(n, k * a)
        jnp = sp.spherical_jn(n, k * a, derivative=True)
        yn = sp.spherical_yn(n, k * a)
        ynp = sp.spherical_yn(n, k * a, derivative=True)
        hn2 = jn + 1j * yn
        hn2p = jnp + 1j * ynp
        bnkra = jn - (jnp / hn2p) * hn2
        for m in range(-n, n+1):
            pnm = shd_nm(channels, n, m) * ((-1)**n) / (bnkra * 4 * np.pi * 1j**n)
            Pnm.append(pnm)
    return Pnm

def Y_nm(n, m):
    em32 = mc.EigenmikeEM32()
    estr = em32.returnAsStruct()
    ths = estr['thetas']
    phs = estr['phis']
    Ynm = np.zeros(32)*1j
    for ind in range(32):
        tq = ths[ind]
        pq = phs[ind]
        Ynm[ind] = sparg_sph_harm(m, n, pq, tq)
    return Ynm

def sparg_sph_harm(m, n, phi, theta):
    sph = (-1)**m * np.sqrt((2*n+1)/(4*np.pi) * sp.factorial(n-np.abs(m)) / sp.factorial(n+np.abs(m))) * sp.lpmn(int(np.abs(m)), int(n), np.cos(theta))[0][-1][-1] * np.exp(1j*m*phi)
    # lpmn ve slicing yerine lpmv kullanabilirmişiz sp.lpmv(n, n, 0)  =  sp.lpmv(n, n, 0)[0][-1][-1], latter is giving a matrix of whole values up to order/degree n.
    return sph

def sparg_sph_harm_list(m, n, phi, theta):
    s = []
    for i in range(len(phi)):
        ph = phi[i]
        th = theta[i]        
        sph = (-1)**m * np.sqrt((2*n+1)/(4*np.pi) * sp.factorial(n-np.abs(m)) / sp.factorial(n+np.abs(m))) * sp.lpmn(int(np.abs(m)), int(n), np.cos(th))[0][-1][-1] * np.exp(1j*m*ph)
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
    sqr_size = (ordd+1)**2 #(Lmax - 1) ** 2
    key, mic_locs = zip(*mics.items())
    Lsize = max(key)
    L_matrix = np.eye(sqr_size*Lsize, dtype=complex)

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
            L_matrix[((row-1) * sqr_size):((row) * sqr_size), ((col-1) * sqr_size):((col) * sqr_size)] = L
    return L_matrix

def D_multipole(C, mics, n, k, a):
    """
    C to D
    """
    size = (n+1)**2
    jhnp = []
    for i in range(n + 1):
        jnp = sp.spherical_jn(i, k * a, derivative=True)
        ynp = sp.spherical_yn(i, k * a, derivative=True)
        hnp = jnp + 1j * ynp
        for i in range((i) * 2 + 1):
            jhnp.append(jnp / hnp)
    jhnp = np.array(jhnp)    
    jhnp_resized = np.resize(jhnp, size*len(mics))
    D_flat = C * -jhnp_resized
    return D_flat

def C_multipole(N, freq, s_sph, k, mics, flag, rot):
    rho = 1.225
    key, mic_locs = zip(*mics.items())
    size = (N+1)**2
    c = 0
    t_size = size*len(mics)
    C_input = np.zeros(t_size)*1j
    c = 0
    phase_list = np.zeros(t_size)*1j
    for keys in itertools.product(mic_locs):
        q = keys[0] 
        rq_p =  q
        if flag == "same cin":
            src_inward_sph = cart2sph(-src[0],-src[1],-src[2]) #r,th,phi 
            src_inward_cart = -src
        elif flag == "pw calibrated cin":
            mic_src = (q - src)
            src_inward_sph = cart2sph(mic_src[0], mic_src[1], mic_src[2]) 
            src_inward_cart = mic_src
        else:
            print("no such flag exists")  
        
        k_vec = k*src_inward_cart
        phase = np.exp(-np.sum(k_vec*rq_p)*1j)        
        for n in range(N+1):
            for m in range(-n, n+1):
                Ynm_s = sparg_sph_harm(m, n, src_inward_sph[2], src_inward_sph[1]).round(10)
                t = c*size + (n+1)**2 - (n-m)
                anm = np.conj(Ynm_s)*np.exp(-1j*rot*m)
                Cnm = anm * 4 * np.pi * (1j)**n * phase / (1j * 2 * np.pi * freq * -rho) 
                C_input[t-1] = Cnm
                phase_list[t-1] = phase
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
    A_nm = LA.lu_solve((lu, piv), D) #Eq.39    
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
    rho = 1.225     # Density of air
    c = 343         # Speed of sound
    em32 = mc.EigenmikeEM32()
    estr = em32.returnAsStruct()
    ths = estr['thetas']
    phs = estr['phis']
    mic32 = np.zeros(32)*1j
    for ind in range(32):
        tq = ths[ind]
        pq = phs[ind]
        potential = 0
        for n in range(0, n_low+1):
            jnp = sp.spherical_jn(n, k * a, derivative=True)
            for m in range(-n,n+1):
                Ynm = sparg_sph_harm(m, n, pq, tq)
                t = (n+1)**2 - (n-m) - 1
                potential += Anm[t]*Ynm / (jnp) #sph_harm(m, n, pw, tw) #Gumerov, Eq. 18
        pressure = -potential * c * rho / (k * a**2)
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
    rho = 1.225     # Density of air
    c = 343         # Speed of sound
    em32 = mc.EigenmikeEM32()
    estr = em32.returnAsStruct()
    ths = estr['thetas']
    phs = estr['phis']
    pressure_all = []
    size = (n_low + 1) ** 2
    for arr in range(no_of_rsmas):
        Anm = Anm_all[arr*size:arr*size+size]
        mic32 = np.zeros(32) * 1j
        for ind in range(32):
            tq = ths[ind]
            pq = phs[ind]
            potential = 0
            for n in range(0, n_low+1):
                jnp = sp.spherical_jn(n, k * a, derivative=True)
                for m in range(-n,n+1):
                    Ynm = sparg_sph_harm(m, n, pq, tq)
                    t = (n+1)**2 - (n-m) - 1
                    potential += Anm[t] * Ynm / (jnp)  #sph_harm(m, n, pw, tw) #Gumerov, Eq. 30
            pressure = -potential * c * rho / (k * (a**2))
            mic32[ind] = pressure
        pressure_all.append(mic32)
    return pressure_all

def shd_nm(channels, n, m):
    em32 = mc.EigenmikeEM32()
    estr = em32.returnAsStruct()
    wts = estr['weights']
    ths = estr['thetas']
    phs = estr['phis']
    pnm = np.zeros(np.shape(channels[0]))*1j
    for ind in range(32):
        cq = channels[ind]
        wq = wts[ind]
        tq = ths[ind]
        pq = phs[ind]
        Ynm = sparg_sph_harm(m, n, pq, tq) #Rafaely Ynm
        pnm += wq * cq * np.conj(Ynm)
    return pnm

def shd_all2(channels, Nmax, k, a):
    Pnm = []
    rho = 1.225
    c = 343
    for n in range(Nmax+1):
        jnp = sp.spherical_jn(n, k * a, derivative=True)
        for m in range(-n, n+1):
            pnm = shd_nm(channels, n, m) * jnp * k * a**2 / (-rho*c)
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

def D_to_Cin(D,mics,jhnp, n):    
    size = (n+1)**2
    jhnp_resized = np.resize(jhnp, size*len(mics))
    C_in_scat = D * (1 / -jhnp_resized)
    return C_in_scat

def C_tilde_to_Anm(C, f, rho, mics):
    w = 2*np.pi*f
    block = int(len(C)/len(mics))
    n = int(np.sqrt(block)-1)
    C_split = C.reshape(len(mics),block)    
    anm = []    
    for row in range(len(C_split)):
        C_single = C_split[row]
        for n in range(0, n+1):
            for m in range(-n, n+1):
                anm.append(C_single*-1j*w*rho/(4*np.pi*((1j)**n)))
    anm = np.array([anm]).flatten()               
    return(anm)

def pfield_sphsparg(f, k, mesh_row, N, C_in):
    """ extrapolated pressure at distance r with given anm's [as a list]"""
    mesh_sph, r = cart2sphr(mesh_row)
    th = mesh_sph[:,0]
    ph = mesh_sph[:,1]
    rho = 1.225
    pr = 0
    kr = k*r
    w = 2*np.pi*f
            
    count = 0
    pr = 0
    for n in range(N+1):
        for m in range(-n,n+1): 
            term = C_in[count]*(-1j*w*rho)*spherical_jn(n, kr, derivative=False)*sparg_sph_harm_list(m, n, ph, th)
            pr = pr + term
            count += 1            
    return(pr)

def pfield_sphsparg_point(f, k, mesh_row, N, C_in):
    """ extrapolated pressure at distance r with given anm's [as a list]"""
    
    r, th, ph = cart2sph_single(mesh_row)
    rho = 1.225
    pr = 0
    kr = k*r
    w = 2*np.pi*f
            
    count = 0
    pr = 0
    for n in range(N+1):
        for m in range(-n,n+1): 
            term = C_in[count]*(-1j*w*rho)*spherical_jn(n, kr, derivative=False)*sparg_sph_harm(m, n, ph, th)
            pr = pr + term
            count += 1            
    return(pr)

def plot_contour_DELAY(pressure, x, vsize):
    """ contour plot (for pressure or angular error) with a 2d meshgrid """
    fig, ax = plt.subplots(figsize=(7, 7))
    fig.gca().set_xlabel(r'$\tau_p\;[ms]$', fontsize=16)
    fig.gca().set_ylabel(r'$\tau_q\;[ms]$', fontsize=16)
            
    l = int(np.sqrt(len(pressure)))
    pressure_shaped = pressure.reshape(l, -1)
    pressure_real = pressure_shaped.real
    
    r_xx,r_yy = np.mgrid[-x:x:(vsize*1j), -x:x:(vsize*1j)]
    # t = plt.contour(r_xx,r_yy, pressure_real, cmap="jet")
    
    CS = ax.contour(r_xx*10**3, r_yy*10**3, pressure_real*10**3, cmap='binary')
    ax.contourf(r_xx*10**3, r_yy*10**3, pressure_real*10**3, cmap='Spectral')  # Wistia  afmhot gist_yarg autumn
    ax.clabel(CS, inline=True, fontsize=18)
    ax.set_aspect("equal")
    
    # ax.set_facecolor('xkcd:salmon')
    # p2 = ax.get_position().get_points().flatten()
    # ax_cbar1 = fig.add_axes([p2[0],p2[2], p2[2]-p2[0], 0.025])
    # plt.colorbar(t,cax=ax_cbar1 ,orientation="horizontal",ticklocation = 'top')
    return(ax)

def plot_scene(src, mics):
    key, mic_locs = zip(*mics.items())
    ad,bd,cd = list(zip(*mic_locs))
    ar,br,cr = list(zip(src))
    fig=plt.figure()
    ax50 = Axes3D(fig)
    ax50.scatter(ad,bd,cd, s=100)
    ax50.scatter(ar,br,cr, c="r", s=250)
    return

def plot_scene_clr(src, mics):
    key, mic_locs = zip(*mics.items())
    ad, bd, cd = list(zip(*mic_locs))
    ar,br,cr = list(zip(src))
    fig=plt.figure()
    ax50 = Axes3D(fig)
    ax50.scatter(ar, br, cr, c="r", s=250)
    for i in range(len(ad)):
        ax50.scatter(ad[i],bd[i],cd[i], s=100)
    return

def rotvec(phis, N, Q):
    q = []
    if not len(phis) == Q:
        pad = Q-len(phis)
        phis = np.pad(phis, (0,pad), 'constant', constant_values=(0))       
    for mic in range(Q):
        for n in range(N+1):
            for m in range(-n, n+1):
                q.append(np.exp(1j*m*phis[mic]))
    return np.array(q)

def rotatemat(MPmat, qrot):
    return qrot @ MPmat 

def cin_roterror_iter(Anm_tilde, L, n):

    o = (n+1)**2
    err = []
    p_rot = np.linspace(0,2*np.pi,360)
    
    for rot in p_rot:
        Anm_tilde_rot = rotate_anm(Anm_tilde, rot, mics)
        D_tilde = Anm_to_D(Anm_tilde_rot, L)
        C_in_tilde = D_to_Cin(D_tilde, mics, jhnp, n)
        
        c1_tilde = C_in_tilde[0:o]
        c2_tilde = C_in_tilde[o:2*o]
        err.append(np.linalg.norm(c1_tilde-c2_tilde))

    err_ = np.array(err)
    ind = err.index(min(err))
    print("order:", n, ", cin rotation iteration:")
    print("min hatanın açısı=", np.degrees(p_rot[ind]), "derece")
    print("max error:", np.max(err_))
    print("min error:", np.min(err_))
    fig=plt.figure()
    plt.plot(p_rot, err)
    plt.title("rotation iteration")
    return

def cin_roterror_iter_3(Anm_tilde, L, n):

    size = (n+1)**2
    err12 = []
    err13 = []
    err23 = []

    p_rot = np.linspace(0,2*np.pi,360)
    
    for rot in p_rot:

        Anm_tilde_rot = rotate_anm(Anm_tilde, rot, mics)
        D_tilde = Anm_to_D(Anm_tilde_rot, L)
        C_in_tilde = D_to_Cin(D_tilde, mics, jhnp, n)
        
        c1_tilde = C_in_tilde[0:size]
        c2_tilde = C_in_tilde[size:2*size]
        c3_tilde = C_in_tilde[2*size:3*size]

        err12.append(np.linalg.norm(c1_tilde-c2_tilde))
        err13.append(np.linalg.norm(c1_tilde-c3_tilde))
        err23.append(np.linalg.norm(c2_tilde-c3_tilde))

    ind12 = err12.index(min(err12))    
    ind13 = err13.index(min(err13))
    ind23 = err23.index(min(err23))

    print("order:", n, ", cin rotation iteration:")
    print("min error for c12 at", np.degrees(p_rot[ind12]), "derece")
    print("min error for c13 at", np.degrees(p_rot[ind13]), "derece")
    print("min error for c23 at", np.degrees(p_rot[ind23]), "derece")

    fig=plt.figure()
    plt.plot(p_rot, err12)
    plt.plot(p_rot, err13)    
    plt.plot(p_rot, err23)
    plt.legend(["|c1-c2|", "|c1-c3|","|c2-c3|"]) 
    plt.title("rotation iteration")
    return

def cin_roterror_3D(Anm_tilde, L, n, p_rot, q_rot):
    
    pX, pY = np.meshgrid(p_rot, q_rot)
    
    size = (n+1)**2
    err12 = []
    err13 = []
    err23 = []
    err_rmse = []
    for prot in p_rot:
        for qrot in q_rot:
            rot_list = [prot, qrot]
            Anm_tilde_rot = rotate_anm_3D(Anm_tilde, rot_list, mics)
            D_tilde = Anm_to_D(Anm_tilde_rot, L)
            C_in_tilde = D_to_Cin(D_tilde, mics, jhnp, n)
        
            c1_tilde = C_in_tilde[0:size]
            c2_tilde = C_in_tilde[size:2*size]
            c3_tilde = C_in_tilde[2*size:3*size]
    
            err12.append(np.linalg.norm(c1_tilde-c2_tilde))
            err13.append(np.linalg.norm(c1_tilde-c3_tilde))
            err23.append(np.linalg.norm(c2_tilde-c3_tilde))              
            
            err_tot = np.linalg.norm(c1_tilde-c2_tilde)**2+np.linalg.norm(c1_tilde-c3_tilde)**2+np.linalg.norm(c2_tilde-c3_tilde)**2
            err_rmse.append(np.sqrt(err_tot/3))            
    return(err_rmse)

def cin_delerror_3D(Anm_tilde, L, n, p_del, q_del):
    
    pX, pY = np.meshgrid(p_del, q_del)    
    size_r = (n+1)**2 # real shd block size      
    index_list = mzeros_index(n)
    size = len(index_list) # size of only m = 0 indexes
    err_rmse = []
    c = 0
    for prot in p_del:
        for qrot in q_del:
            c += 1
            delay_list = [prot, qrot, 0]
            Anm_tilde_del = delay_anm_3D(Anm_tilde, delay_list, mics, n,f)
            D_tilde = Anm_to_D(Anm_tilde_del, L)
            C_in_tilde = D_to_Cin(D_tilde, mics, jhnp, n)        
        
            cr = []
            for t in range(len(mics)):
                for ind in index_list:                
                    cr.append(C_in_tilde[ind+(t*size_r):ind+(t*size_r)+1])
                    
            C_in_tilde_reduced = np.array(cr).flatten()            
            comb = combinations(range(1, len(mics)+1), 2)  
            err_tot = 0  
            
            for i,j in list(comb):  
                term = np.linalg.norm(C_in_tilde_reduced[(i-1)*size:i*size]-C_in_tilde_reduced[(j-1)*size:j*size])**2
                err_tot = err_tot + term
                    
            err_tot = (np.sqrt(err_tot/len(mics))) 
            err_rmse.append(err_tot)  
    return(err_rmse)

def clamp(a, amax):
    if a > amax:
        return amax
    else:
        return a

def ordervec(fr, ra, Nmax):
    c=341.
    fcnt = len(fr)
    kra = np.abs(2 * np.pi * fr / c * ra)
    orderlist = []
    for find in range(fcnt):
        krai = kra[find]
        orderlist.append(int(clamp(np.round(krai), Nmax)))
    return orderlist

def jhnp_func(n,k,a):
    jhnp = []
    for i in range(n + 1):
        jnp = sp.spherical_jn(i, k * a, derivative=True)
        ynp = sp.spherical_yn(i, k * a, derivative=True)
        hnp = jnp + 1j * ynp
        for i in range((i) * 2 + 1):
            jhnp.append(jnp / hnp)
    jhnp = np.array(jhnp)   
    return(jhnp)

def rotate_anm(Anm_tilde, azi_rot, mics):
    Anm_tilde_diag = np.diag(Anm_tilde)
    qr = rotvec([azi_rot], n, len(mics))
    Anm_tilde_rot = rotatemat(Anm_tilde_diag, qr)     
    return(Anm_tilde_rot)
         
def rotate_anm_3D(Anm_tilde, rot_list, mics):
    Anm_tilde_diag = np.diag(Anm_tilde)
    qr = rotvec(rot_list, n, len(mics))
    Anm_tilde_rot = rotatemat(Anm_tilde_diag, qr)     
    return(Anm_tilde_rot)

def delay_anm_3D(Anm_tilde, delay_list, mics, order, f):
    
    if not len(delay_list) == len(mics):
        pad = len(mics)-len(delay_list)
        delay_list = np.pad(delay_list, (0,pad), 'constant', constant_values=(0))
    w = 2*np.pi*f
    Anm_tilde_diag = np.diag(Anm_tilde)        
    q = []
    for mic in range(len(mics)):
        for n in range(order+1):
            for m in range(-n, n+1):
                q.append(np.exp(1j*w*delay_list[mic]))
    Anm_tilde_delayed = Anm_tilde_diag @ q
    return(Anm_tilde_delayed)
       
def find_rot(Anm_tilde_rot, true_azi_rot, n):
    """ rotates anm iteratively and finds the minimum error between two cin's (two rsmas) """
    o = (n+1)**2 
    p_rot = np.linspace(0,2*np.pi,360)
    err_r_r = []
    for rot in p_rot:     
        Anm_tilde_r_r = rotate_anm(Anm_tilde_rot, rot, mics)   
        D_tilde_r = Anm_to_D(Anm_tilde_r_r, L)
        C_in_tilde_r = D_to_Cin(D_tilde_r, mics, jhnp, n)
        c1_tilde_r_r = C_in_tilde_r[0:o]
        c2_tilde_r_r = C_in_tilde_r[o:2*o]
        err_r_r.append(np.linalg.norm(c1_tilde_r_r-c2_tilde_r_r))
    
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
    o = (n+1)**2 
    p_rot = np.linspace(0,2*np.pi,360)
    err_r_r12 = []
    err_r_r13 = []
    err_r_r23 = []
    for rot in p_rot:     
        Anm_tilde_r_r = rotate_anm(Anm_tilde_rot, rot, mics)   
        D_tilde_r = Anm_to_D(Anm_tilde_r_r, L)
        C_in_tilde_r = D_to_Cin(D_tilde_r, mics, jhnp, n)
        c1_tilde_r_r = C_in_tilde_r[0:o]
        c2_tilde_r_r = C_in_tilde_r[o:2*o]
        c3_tilde_r_r = C_in_tilde_r[2*o:3*o]

        err_r_r12.append(np.linalg.norm(c1_tilde_r_r-c2_tilde_r_r))
        err_r_r13.append(np.linalg.norm(c1_tilde_r_r-c3_tilde_r_r))
        err_r_r23.append(np.linalg.norm(c2_tilde_r_r-c3_tilde_r_r))
    
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
    print("min error for c12:", np.min(err23_), "at index:", ind_min23, "angle [degree]:", np.degrees(p_rot[ind_min23]), "\n")

    plt.figure()
    plt.plot(p_rot, err_r_r12) 
    plt.plot(p_rot, err_r_r13)  
    plt.plot(p_rot, err_r_r23)  
    plt.legend(["|c1-c2|", "|c1-c3|","|c2-c3|"]) 
    plt.title("finding rotation")
    return

def CDLADC_tilde(C_in, mics, n, k, a, jhnp, SNR): # L'yi soldan R-1 ile çarpabiliriz
    """ main calculations of gumerovs + spargs algorithm
    input: c_in for plane wave
    output: c_in_tilde according to eigenmic configuration """
    key, _ = zip(*mics.items())
    no_of_rsmas = len(key)
    
    D  = D_multipole(C_in,mics,n,k,a)
    L = L_multipole(n, a, k, mics)
    _, Anm_all = A_multipole(L, D, n)
    presmulti = pressure_withA_multipole(n, a, k, Anm_all, no_of_rsmas)
    presmulti_n = add_noise(presmulti, SNR, no_of_rsmas)
    Anm_tilde = pressure_to_Anm(presmulti_n, n, no_of_rsmas,k,a)
    D_tilde = Anm_to_D(Anm_tilde, L)
    C_in_tilde = D_to_Cin(D_tilde, mics,jhnp,n)    
    return(C_in_tilde, Anm_tilde, L)

def Anmreal_cin_tilde(anm, mics, n, k, a, jhnp):

    L = L_multipole(n, a, k, mics)
    D_tilde = Anm_to_D(anm, L)
    C_in_tilde = D_to_Cin(D_tilde, mics, jhnp, n)

    return(C_in_tilde, L)


def ADC_tilde(Anm, L, jhnp, mics, n):
    """ input: anm_tilde output: c_tilde """
    D_tilde = Anm_to_D(Anm, L)
    C_in_tilde = D_to_Cin(D_tilde, mics, jhnp, n)
    return(C_in_tilde)

def add_noise(pressure, SNR, no_of_poles):
    """
    :param mic32: pressure at each mic
    :param SNR: Signal-to-Noise ratio (dB)
    :return:
    """
    for i in range(no_of_poles):
        pres_temp = pressure[i]
        noise = np.random.rand(32,)
        noise_r = np.random.randn(32)
        noise_i = np.random.randn(32)
        noise_r = noise_r - np.mean(noise_r)
        noise_i = noise_i - np.mean(noise_i)
        noise = noise_r + noise_i * 1j
        mic_norm = np.linalg.norm(pres_temp, axis= 0)
        noisy_pres = 0
        noise_norm = np.linalg.norm(noise, axis=0)
        coef = mic_norm/noise_norm
        SNR_linear = 10**(-SNR/20)
        noise = noise * coef * SNR_linear
        pressure[i] = pres_temp + noise
    noisy_pres = pressure
    return noisy_pres

def cin_scaling(mics, src, N):
    key, mic_locs = zip(*mics.items())    
    mic_src_list = []
    size = (N+1)**2
    for keys in itertools.product(mic_locs):
        mic = keys[0]    
        mic_src = mic - src
        mic_src_list.append(-mic_src)
        
    mic_src_sph = cart2sphr_sparg(np.array(mic_src_list))
    ynmz = []
    for n in range(N+1):
        for m in range(-n, n+1):
            Ynm_s = sparg_sph_harm_list(m, n, mic_src_sph[2], mic_src_sph[1])
            # Ynm_s = sph_harm(m, n, mic_src_sph[:,1], mic_src_sph[:,0])
            ynmz.append(np.conj(Ynm_s))
    ynm = np.array(ynmz).reshape(size, len(mics))
    return(ynm, mic_src_sph)

def total_cin_err(rot_list, Anm_tilde_rotated, L, n, mics):
      
    ynm_scale, _ = cin_scaling(mics, src, n)
    rot_list = np.array(rot_list)
    size = (n+1)**2
    Anm_tilde_rot_rot = rotate_anm_3D(Anm_tilde_rotated, rot_list, mics)
    C_in_tilde = ADC_tilde(Anm_tilde_rot_rot, L, jhnp, mics, n)
    
    comb = combinations(range(1, len(mics)+1), 2)  
    err_sum = 0
    for i,j in list(comb):         
        term = np.linalg.norm(C_in_tilde[(i-1)*size:i*size]*ynm_scale[:, (j-1)].round(10)-C_in_tilde[(j-1)*size:j*size]*ynm_scale[:, (i-1)].round(10))**2
        err_sum = err_sum + term
    err_rmse = (np.sqrt(err_sum/len(mics)))  
    return(err_rmse)


def total_cin_err_real(rot_list, anm_time_aligned, Larr, n, mics, f, ind, a):
    # single frequency optimisation for rotation (real signal)
    k = 2 * pi * f / c
    jhnp = jhnp_func(n, k, a)

    ynm_scale, _ = cin_scaling(mics, src, n)
    rot_list = np.array(rot_list)
    size = (n + 1) ** 2
    Anm_tilde_rot_rot = rotate_anm_3D(anm_time_aligned, rot_list, mics)
    C_in_tilde = ADC_tilde(Anm_tilde_rot_rot, Larr[ind], jhnp, mics, n)

    comb = combinations(range(1, len(mics) + 1), 2)
    err_sum = 0
    for i, j in list(comb):
        term = np.linalg.norm(C_in_tilde[(i - 1) * size:i * size] * ynm_scale[:, (j - 1)].round(10) - C_in_tilde[(j - 1) * size:j * size] * ynm_scale[:,(i - 1)].round(10)) ** 2
        err_sum = err_sum + term
    err_rmse = (np.sqrt(err_sum / len(mics)))
    return (err_rmse)

def total_cin_roterr_fiter_real(rot_list, anm_time_aligned, Larr, n, mics, farr, a):
    # multiple frequency optimisation for rotation (real signal)

    rw = len(farr)
    err_rsme = 0
    ynm_scale, _ = cin_scaling(mics, src, n)
    size = (n + 1) ** 2
    for ind in range(rw):
        f = farr[ind]
        k = 2 * pi * f / c
        jhnp = jhnp_func(n, k, a)

        rot_list = np.array(rot_list)
        Anm_tilde_rot_rot = rotate_anm_3D(anm_time_aligned[:,ind], rot_list, mics)
        C_in_tilde = ADC_tilde(Anm_tilde_rot_rot, Larr[ind], jhnp, mics, n)
        # C_in_tilde = Anm_tilde_rot_rot
        comb = combinations(range(1, len(mics) + 1), 2)
        err_sum = 0
        for i, j in list(comb):
            term = np.linalg.norm(C_in_tilde[(i - 1) * size:i * size] * ynm_scale[:, (j - 1)].round(10) - C_in_tilde[(j - 1) * size:j * size] * ynm_scale[:,(i - 1)].round(10)) ** 2
            err_sum = err_sum + term

        err_rsme = err_rsme + (np.sqrt(err_sum / len(mics)))
    return (err_rsme)

def mzeros_index(order):
    index = 0
    ind = []
    for n in range(order+1):
        for m in range(-n, n+1):            
            if m==0:
                ind.append(index)
            index += 1
    return(ind)

def total_cin_delayerr(delay_list, Anm_tilde_delayed, L, n, mics,f):
    delay_list = np.array(delay_list)
    size_r = (n+1)**2 # real shd block size   
    
    index_list = mzeros_index(n)
    size = len(index_list) # size of only m = 0 indexes

    Anm_tilde_del_del = delay_anm_3D(Anm_tilde_delayed, delay_list, mics, n,f)
    C_in_tilde = ADC_tilde(Anm_tilde_del_del, L, jhnp, mics, n)
    cr = []
    for t in range(len(mics)):
        for ind in index_list:                
            cr.append(C_in_tilde[ind+(t*size_r):ind+(t*size_r)+1])
            
    C_in_tilde_reduced = np.array(cr).flatten()
    comb = combinations(range(1, len(mics)+1), 2)  
    err_sum = 0
    
    for i,j in list(comb):  
        term = np.linalg.norm(C_in_tilde_reduced[(i-1)*size:i*size]-C_in_tilde_reduced[(j-1)*size:j*size])**2
        err_sum = err_sum + term
            
    err_rmse = (np.sqrt(err_sum/len(mics)))  
    return(err_rmse)

def total_cin_delayerr_fiter(delay_list, Anm_tilde_delayed, L, n, mics, nf, fmin, fmax):
    farr = np.linspace(fmin, fmax, nf)
    
    err_rsme = 0 
    for f in farr:        
        delay_list = np.array(delay_list)
        size_r = (n+1)**2 # real shd block size   
        
        index_list = mzeros_index(n)
        size = len(index_list) # size of only m = 0 indexes
    
        Anm_tilde_del_del = delay_anm_3D(Anm_tilde_delayed, delay_list, mics, n,f)
        C_in_tilde = ADC_tilde(Anm_tilde_del_del, L, jhnp, mics, n)
        cr = []
        for t in range(len(mics)):
            for ind in index_list:                
                cr.append(C_in_tilde[ind+(t*size_r):ind+(t*size_r)+1])
                
        C_in_tilde_reduced = np.array(cr).flatten()
        comb = combinations(range(1, len(mics)+1), 2)  
        err_sum = 0
        
        for i,j in list(comb):  
            term = np.linalg.norm(C_in_tilde_reduced[(i-1)*size:i*size]-C_in_tilde_reduced[(j-1)*size:j*size])**2
            err_sum = err_sum + term
                
        err_rsme = err_rsme + (np.sqrt(err_sum/len(mics)))  
    return(err_rsme)

def total_cin_delayerr_fiter_real(delay_list, Anm_tilde_delayed, Larr, n, mics, farr, a):
    rw = len(farr)
    err_rsme = 0

    for ind in range(rw):
        f = farr[ind]
        k = 2 * pi * f / c
        jhnp = jhnp_func(n,k,a)
        delay_list = np.array(delay_list)
        size_r = (n + 1) ** 2  # real shd block size

        index_list = mzeros_index(n)
        size = len(index_list)  # size of only m = 0 indexes
        Anm_tilde_del_del = delay_anm_3D(Anm_tilde_delayed[:,ind], delay_list, mics, n, f)
        C_in_tilde = ADC_tilde(Anm_tilde_del_del, Larr[ind], jhnp, mics, n)
        # C_in_tilde = Anm_tilde_del_del
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

"""def tri(area): 
    center_x, center_y  = 0 , 0
    z = 0
    side = np.sqrt((area*4/np.sqrt(3)))
    x,y = center_x-side/2 , center_y-side*np.sqrt(3)/6
    points_t = tuple([ (x,y,z), (x+side,y,z), (x+side/2,y+side*np.sqrt(3)/2,z)])
    return points_t
 
def sq(area):
    center_x, center_y  = 0 , 0
    z = 0
    h = np.sqrt(area) # height
    w = h # width
    x, y = center_x, center_y 
    points_s = tuple([(x+w/2,y+h/2,z), (x+w/2,y-h/2,z), (x-w/2,y-h/2,z), (x-w/2,y+h/2,z)])
    return points_s
 
def three():
    return "March"
 
def grid_points(argument, area):
    switcher = {"tri": tri(area), "square": sq(area), 3: three}
    # Get the function from switcher dictionary
    return(switcher.get(argument, lambda: "Invalid"))

class Canvas:
    
    
    # Top-Left position in the canvas is (0,0), Bottom-Right position is (width,height)
    # Any point in the canvas is represented by (x,y) value where 0 <= x <= width, 0 <= y <= height
    
    
    GRID_SIZE = 23
    d = GRID_SIZE
    
    def __init__(self, width, height, color="black"):
        self.width = width
        self.height = height
        self.color = color
        self.fig, self.ax = plt.subplots()
        self.ax.get_xaxis().set_visible(False)
        self.ax.get_yaxis().set_visible(False)
        self.ax.add_patch(plt.Rectangle((0,0), width+20, height+20, ec = "black", fc="white", fill=True))

    def _convert_xy(self,x, y):
        return (x,  self.height-y)
    
    def rect(self):       
        rect_cord = []

        for x in range(Canvas.d // 2, self. width, Canvas.d):    
            for y in range(Canvas.d // 2,  self.height, Canvas.d):        
                rect_cord.append(canvas._convert_xy(x,y))
                self.ax.add_patch(plt.Circle((x,y), 0.5, fc="gray"))        

    def tri(self):  
        tri_cord = []
        tri_cordd = []
        z = 0
        count = 0
        for x in range(Canvas.d // 2,  self.width, Canvas.d):    
            for y in range(Canvas.d // 2,  self.height, Canvas.d):          
                tri_cord.append(canvas._convert_xy(x,y))
                c2 = 0
                if count%2 != 0:
                    c2 += 1
                    self.ax.add_patch(plt.Circle((x+Canvas.d/2,y), 0.5, fc="blue"))
                    tri_cordd.append((x+Canvas.d/2,y,z))
                else:
                    self.ax.add_patch(plt.Circle((x,y), 0.5, fc="gray"))
                    tri_cordd.append((x,y,z))
                count += 1
        return(tri_cordd)
    def hexa(self):
        hex_cord = []
        count = 0
        for y in range(Canvas.d // 2,  self.height, Canvas.d):          
            c2 = 1
            if (count%2)==0:
                for x in range(Canvas.d // 2,  self.width, Canvas.d):            
                    if (c2+1)%3 != 0:                               
                        hex_cord.append(canvas._convert_xy(x,y))
                        self.ax.add_patch(plt.Circle((x+Canvas.d/2,y), 0.5, fc="blue"))
                    else:
                        pass
                    c2 += 1
            else:
                for x in range(Canvas.d // 2,  self.width, Canvas.d):            
                    if (c2+1)%3 != 0:                               
                        hex_cord.append(canvas._convert_xy(x,y))
                        self.ax.add_patch(plt.Circle((x+2*Canvas.d,y), 0.5, fc="blue"))
                    else:
                        pass
                    c2 += 1    
            count += 1
            
    def draw(self):
        plt.axis('equal')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

class Hex(object):
    
    def __init__(self, radius, center):
        self._radius = radius
        self._centerx, self._centery, self._centerz = center

    def get_points(self):
        p = []
        for i in range(6):
            angle_deg = 60 * i - 30
            angle_rad = np.radians(angle_deg)
            point = self._centerx + self._radius * np.cos(angle_rad), self._centery + self._radius * np.sin(angle_rad), self._centerz
            p.append(point)            
        pr = np.array(p).round(3)
        return(pr)
    
    def combine(h1, h2):
        p1 = h1.get_points()
        p2 = h2.get_points()
        pall = np.concatenate([p1,p2])
        p = np.unique(pall,axis=0)
        p_set = set(map(tuple, map(tuple, pall)))
        return(p_set, p)
"""

def toa(mics, src, fs, c=341.):
    toa_list = []
    key, mic_locs = zip(*mics.items())
    for i in range(len(mic_locs)):
        toa_list.append(np.linalg.norm(src-mic_locs[i])*fs/c)
    print("toa_sample=", toa_list)
    return(toa_list)

def rawread(wave_file_dir):
    rate, raw = wavfile.read(wave_file_dir, mmap=False)
    row, col = np.shape(raw)
    numchan = col
    numlen = row
    return numlen, numchan, rate, raw

def raw2ambix(ref_data):
    #rate1, raw = wavfile.read(filepos, mmap=False)
    irdir = './data/A2B-Zylia-3E-Jul2020.wav'
    raw = ref_data
    rate2, eir = wavfile.read(irdir, mmap=False)
    #assert rate1 == rate2 # Make sure that the sampling rates are the same
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
    return ambix

def fdambix(ambixchans,nfft):
    #Ambix to Frequency domain translation
    # print(ambixchans)
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

def listoflists(array):
    rw, cl = np.shape(array)

    samples = []
    ls = list(np.transpose(array))
    for ch in range(cl):
        samples.append(list(ls[ch]))
    return samples

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

def wav2shd(subdir, irdir):
    shd_list = []
    dr = os.path.join(".\data", subdir)
    for filedir in glob.glob(dr):
        print(filedir)
        rate, abx = amb.raw2ambix(filedir, irdir)
        fd = amb.fdambix(abx, 1024)
        fda = amb.fdlist2array(fd)
        fdsh = amb.ambix2sh(fda)
        shd_list.append(fdsh)
    return rate, shd_list

def wav2shd_nmax(subdir, irdir, nmax):
    shd_list = []
    raw_list = []
    dr = os.path.join(".\data", subdir)
    for filedir in glob.glob(dr):
        print(filedir)
        rate, abx, raw = amb.raw2ambix(filedir, irdir)
        raw_list.append(raw)
        fd = amb.fdambix(abx, 1024)
        fda = amb.fdlist2array(fd)
        fdsh = amb.ambix2sh(fda)
        shd_list.append(fdsh[:, :(nmax+1)**2])
    return shd_list, raw_list

def wav2shd_nmax_pre(subdir, irdir, nmax, mics, src, fs):
    shd_list = []
    offset = -100
    raw_list = []
    toa_list = toa(mics, src, fs, c=341.)
    ind = 0
    dr = os.path.join("./data", subdir)
    for filedir in glob.glob(dr):
        print("====___")
        print(filedir)
        rate, abx, raw = amb.raw2ambix(filedir, irdir)

        rawa = align(raw, toa_list[ind], offset)
        abxa = align(abx, toa_list[ind], offset)
        raw_list.append(rawa)
        ind += 1

        fd = amb.fdambix(abxa, 1024)
        fda = amb.fdlist2array(fd)
        fdsh = amb.ambix2sh(fda)
        shd_list.append(fdsh[:, :(nmax + 1) ** 2])
    return shd_list, raw_list

def align(input, toa, offset=0):
    assert toa>-offset
    print(input.shape)
    out = input[int(toa+offset):,:]
    return out

def stackshd(shd_list, f_ind, n0):
    # stack mic shd's
    b = np.zeros((len(f_ind), (n0+1)**2))
    # print(len(shd_list))
    for mic in shd_list:
        a = mic[f_ind, :]
        b = np.append(b, a, axis=1)
        anm = np.transpose(b[:, (n0 + 1) ** 2:])
        # print(anm.shape)
    return anm

def shift(raw, shift):
    _, _, _, raw0 = rawread(raw)
    raw0 = np.transpose(raw0)
    if shift < 0:
        shift = -shift
        raw0_shifted = np.pad(raw0, ((0, 0), (0, shift)), constant_values=0)
    else:
        raw0_shifted = np.pad(raw0, ((0, 0), (shift, 0)), constant_values=0)
    return(raw0_shifted)

if __name__=='__main__':

    fs = 48000
    pi = np.pi
    n = 3 # Spherical Harmonic Order
    a = 5.5e-2  # Radius of sphere  (42e-3 for eigenmic)
    fft_block_size = 512


    """ Load Mic Locations """

    # distance = meters
    # direct_30cm_mic1
    # m1 = np.array([-2.51038350e-04,  4.74460295e-01, 0])
    # m2 = np.array([-3.01130053e-01,  4.38673276e-01, 0])
    # src = np.array([3.01381091e-01, -9.13133571e-01, 0])
    #
    # # direct_60cm_mic1
    # m1 = np.array([0.02005751, 0.48298869, 0])
    # m2 = np.array([-0.56855144, 0.3520349, 0])
    # src = np.array([0.54849393, -0.83502359, 0])
    #
    # # ang3
    # m1 = np.array([0.14549194, 0.68440495, 0])
    # m2 = np.array([-0.69286712, 0.00355691, 0])
    # src = np.array([0.54737518, -0.68796186, 0])

    # mics = {1: m1, 2: m2}

    mic_list_z, mic_dict_z = json2miclist_z("./data/largehexalab.json")

    # z axis of seven mics set to 0
    mic_list_z[0:7, 2] = 0.885
    # z axis of the source set to 1.3
    mic_list_z[7, 2] = 1.3

    mic_dict_z = list2dict(mic_list_z[0:7, :])
    # mic_dict_z = list2dict(mic_list_z[0:2, :])
    mics = mic_dict_z

    src = mic_list_z[7]
    rsrc_sph = cart2sph(src[0], src[1], src[2]) # source coordinates in spherical coors.

    """ Ambixutil: wav-ambix-shd conversion """

    irdir = './data/A2B-Zylia-3E-Jul2020.wav'
    # subdir = r"smallhexa_lab\30deg_IR\*"
    # subdir = r"bighexa_lab\reference_IR\*"
    subdir = r"./bighexa_lab/reference_no_reflections/*"

    # rate, shd_list = wav2shd(subdir, irdir)
    shd_list, wav_list2 = wav2shd_nmax(subdir, irdir, nmax=n) # for different n's
    shd_list, wav_list = wav2shd_nmax_pre(subdir, irdir, n, mics, src, fs)  # for different n's, geo delay prealigned

    for raw in wav_list:
        plt.plot(raw[0,:])

    plt.show()

    for raw in wav_list2:
        plt.plot(raw[0,:])

    plt.show()

    # filedir1 = './data/directpath/rotation/ir0_norm_ang3.wav'
    # filedir2 = './data/directpath/rotation/ir1_norm_ang3.wav'
    # rate, abx1 = amb.raw2ambix(filedir1, irdir)
    # rate, abx2 = amb.raw2ambix(filedir2, irdir)
    # samples = amb.listoflists(abx1)
    # fd1 = amb.fdambix(abx1, 1024)
    # fd2 = amb.fdambix(abx2, 1024)
    # fda1 = amb.fdlist2array(fd1)
    # fda2 = amb.fdlist2array(fd2)
    # fdsh1 = amb.ambix2sh(fda1)
    # fdsh2 = amb.ambix2sh(fda2)

    """ Main ALgo. """
    ### file LOAD
    # file = open('direct_60cm_mic1', "rb")
    # data1 = pickle.load(file)
    # file.close()
    # file = open('direct_60cm_mic2', "rb")
    # data2 = pickle.load(file)
    # file.close()

    # file = open('direct_ang3_mic1', "rb")
    # data1 = pickle.load(file)
    # file.close()
    # file = open('direct_ang3_mic2', "rb")
    # data2 = pickle.load(file)
    # file.close()

    f_list = np.linspace(0, 48000/2, len(shd_list[0]))
    # fr = np.array([1000,2000,3000])
    n_list = ordervec(f_list, 5.5e-2, 3)
    fr_ind = fs/2/fft_block_size
    print("freq step size=", fr_ind)
# kra<1,2,3

    if n==0:
        flow = 1
        fhigh = 652/fr_ind
    elif n==1:
        flow = 652/fr_ind
        fhigh = 1301/fr_ind
    elif n==2:
        flow = 1301/fr_ind
        fhigh = 2607/fr_ind
    elif n==3:
        flow = 2607/fr_ind
        fhigh = 300 # upper_ind=300 since >20kHz was noisy

    # flow = 55
    # fhigh = 300
    np.random.seed(250)
    f_ind = np.random.randint(int(flow), int(fhigh), size=5) #random seed = 15 ve flow= 55'da çalışmıyor???
    # f_ind = np.array([31, 38, 41, 45, 51])
    # f_ind = np.array([31, 51])
    fr = f_list[np.unique(f_ind)]
    fr = f_list[f_ind]

    # fr = np.array([1371])

    # d1 = data1[f_ind, :]
    # d2 = data2[f_ind, :]
    # dtot = np.hstack((d1,d2))
    # anm = np.transpose(dtot)

    """ stack mic shd's """

    anm = stackshd(shd_list, f_ind, n0=n)

    dellist = []
    er_del_square = np.zeros((90,90))
    size = (n+1)**2

    c = 343     #Speed of sound
    order = n  # Spherical Harmonic Order for L matrix
    print("\norder=", n)
    del_err = dict()
    del_error_list = []

    """ Main """
    # numeric simulation (r11)
    # flags = ("same cin", "pw calibrated cin")
    # C_in_phase, phaselist = C_multipole(n, f, rsrc_sph, k, mics, flag = flags[1], rot=0) # "same cin" or "pw calibrated cin"
    # C_in = C_in_phase/phaselist
    # C_in_tilde, Anm_tilde, L = CDLADC_tilde(C_in, mics, n, k, a, jhnp, SNR)
    # C_in_tilde, L = Anmreal_cin_tilde(anm, mics, n, k, a, jhnp)
    Anm_tilde_delayed = anm

    Larr = []
    for ind in range(len(fr)):
        k = 2 * pi * fr[ind] / c
        Larr.append(L_multipole(n, a, k, mics))

    print("\nno of rsma's =", len(mics), "fr =", fr)

    # %% """ TIME DELAY CALIBRATION """
    #numeric simulation (r11)
    # sample32 = 0.3/f
    # delay_list = [0.0001, 0.00015]
    # delay_list = ((0.9*np.random.random_sample((len(mics)-1,))+0.1)*sample32)
    # if len(delay_list) == len(mics): print("input delay list should be less than # of mics")
    # print("\ntrue delay list =", delay_list)
    # Anm_tilde_delayed = delay_anm_3D(Anm_tilde, delay_list, mics, order, f)
    # bnd = sample32

    del_bnd = 2.67e-3 * 4  #
    # del_bnd = 2.67e-3 *
    # p_del = np.linspace(-del_bnd, del_bnd, 90)
    # q_del = np.linspace(-del_bnd, del_bnd, 90)
    # pX, qY = np.meshgrid(p_del, q_del)
    #
    # del_no = 200
    # del_r = np.linspace(-del_bnd,del_bnd,del_no)
    # err_list = []

    # for ind in range(del_no):
    #     err_list.append(total_cin_delayerr_fiter_real([del_r[ind]], anm, Larr, n, mics, np.array(fr), a))
    #
    # plt.plot(del_r, np.array(err_list))
    # plt.show()

    if len(mics)==3:
        print("contour printed")
        er = np.array(cin_delerror_3D(Anm_tilde_delayed, L, n, p_del, q_del))
        er_del_square += er.reshape(len(p_del), len(q_del))

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(pX, qY, er_del_square)

        fig.colorbar(surf, shrink=0.5, aspect=5)

        plot_contour_DELAY(er, bnd, 90)
        plt.plot(-delay_list[0]*10**3, -delay_list[1]*10**3, 'wo')

        # plt.title(" for N=%i f=%i, " %(n,f))

    # bnds = tuple([(-del_bnd, del_bnd)] * (len(mics)))
    bnds = tuple([(-del_bnd, del_bnd)] * (len(mics)-1))
    x0 = np.zeros(len(mics))
    # x0 = np.zeros(len(mics) - 1)
    rmse_ = []
    sim_no = 1
    for epoch in range(sim_no):
        print("\nsimulation number:", epoch)

        " BASINHOPPING WITH LBFGSB"
        # res = minimize(total_cin_delayerr_fiter_real, x0, (Anm_tilde_delayed, Larr, n, mics, fr, a), method='L-BFGS-B', bounds=bnds)
        # print("minimum with only l-bfgs-b:", res.x,  "\nf(x) = %.9f" % (res.fun))
        # minimizer_kwargs = {"method":"L-BFGS-B", "args": (Anm_tilde_delayed, Larr, n, mics, fr, a), "bounds":bnds, "jac":False}
        # ret = basinhopping(total_cin_delayerr_fiter_real, x0, minimizer_kwargs=minimizer_kwargs, niter=100)
        # print("minimum with basinhopping: x = ", ret.x,  "\nf(x) = %.9f" % (ret.fun))        #
        # delay = ret.x
        # print("delay sample:\n", 48000 * np.array([delay]))
        # dellist.append([(x*f+np.arange(-10,10))/f for x in res.x])

        " DIFFERENTIAL EVOLUTION"
        # params = (Anm_tilde_delayed, Larr, n, mics, fr, a)
        # ret = brute(total_cin_delayerr_fiter_real, [(-del_bnd, del_bnd)], args=params, full_output=True)
        # ret = brute(total_cin_delayerr_fiter_real, bnds, args=params, full_output=True)
        # print("minimum with brute force: x = ", ret)  #
        # delay = ret.x
        # print("delay sample:\n", 48000 * np.array([ret]))
        params = (Anm_tilde_delayed, Larr, n, mics, fr, a)
        res = differential_evolution(total_cin_delayerr_fiter_real, bnds, args=params) #[(-del_bnd, del_bnd)]
        delay = res.x

        delay_as_sample = 48000*np.array([delay])
        print("delay:", delay)
        print("delay sample:", delay_as_sample)

    sample_shift = delay[0] * fs
    sample_shift = round(sample_shift) #int(sample_shift) -> +1sample

    print("data1 is %d sample ahead the data2" %(-sample_shift))

    # print("delays-toa:", delay_as_sample - toa_list)

    _, _, _, raw0 = rawread("ir0_norm_ang3.wav")
    _, _, _, raw1 = rawread("ir1_norm_ang3.wav")
    raw0 = np.transpose(raw0)
    raw1 = np.transpose(raw1)

    if sample_shift<0:
        sample_shift = -sample_shift
        data1_shifted = np.pad(raw0, ((0, 0), (sample_shift, 0)), constant_values=0)
        data2_shifted = np.pad(raw1, ((0, 0), (0, sample_shift)), constant_values=0)

    else:
        data1_shifted = np.pad(raw0, ((0, 0), (0, sample_shift)), constant_values=0)
        data2_shifted = np.pad(raw1, ((0, 0), (sample_shift, 0)), constant_values=0)

    ### Original and Time-aligned Signal Plots
    # """
    plt.figure()
    plt.plot(raw0[0])
    plt.plot(raw1[0])
    plt.show()

    plt.figure()
    plt.plot(data1_shifted[0])
    plt.plot(data2_shifted[0])
    plt.show()
    # """

    """GET SHD OF SHIFTED DATA"""
    #RAW2AMBIX
    abx_1 = raw2ambix(data1_shifted)
    abx_2 = raw2ambix(data2_shifted)

    #AMBIX2SHD
    fd1 = fdambix(abx_1, 1024)
    fd2 = fdambix(abx_2, 1024)
    fda1 = fdlist2array(fd1)
    fda2 = fdlist2array(fd2)
    fdsh1 = ambix2sh(fda1)
    fdsh2 = ambix2sh(fda2)

    d1 = fdsh1[f_ind, :]
    d2 = fdsh2[f_ind, :]

    dtot = np.hstack((d1,d2))
    anm_time_aligned = np.transpose(dtot)

    rot_bnd = np.pi
    bnds = tuple([(-rot_bnd, rot_bnd)] * (len(mics)))

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
    minimizer_kwargs = {"method": "Nelder-Mead", "args": (anm_time_aligned, Larr, n, mics, fr, a), "bounds": bnds, "jac": False}
    ret = basinhopping(total_cin_roterr_fiter_real, np.zeros(len(mics)), minimizer_kwargs=minimizer_kwargs, niter=50)
    print("global minimum with basinhopping: x = ", np.degrees(ret.x), "\nf(x) = %.4f" % (ret.fun))

    params = (anm_time_aligned, Larr, n, mics, fr, a)
    res = differential_evolution(total_cin_roterr_fiter_real, [(-rot_bnd, rot_bnd), (-rot_bnd, rot_bnd)], args=params)
    print("global minimum with differential_evolution: x = ", np.degrees(res.x), "\nsuccess:", res.success, "\nres.fun:", res.fun)

    #Surface plot of rotation error for 2 mics
    p_del = np.linspace(-rot_bnd, rot_bnd, 90)
    q_del = np.linspace(-rot_bnd, rot_bnd, 90)
    pX, qY = np.meshgrid(p_del, q_del)
    err_mat = np.zeros((90, 90))

    for ind in range(90):
        for jnd in range(90):
            # for single frequency:
            # err_mat[ind, jnd] = total_cin_err_real(np.array([pX[ind, jnd], qY[ind, jnd]]), anm_time_aligned, Larr, n,
            # mics, f, f_ind, a)

            # for multiple frequency:
            err_mat[ind, jnd] = total_cin_roterr_fiter_real(np.array([pX[ind, jnd], qY[ind, jnd]]), anm_time_aligned,
                                                            Larr, n, mics, fr, a)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(pX, qY, err_mat, cmap="jet")
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.title("Surface Plot")
    plt.show()


