import numpy as np
import healpy as hp
import scipy.special as spec
from scipy.special import legendre, eval_legendre#, sph_jnyn
# from ompseparate import *

def sph2cart(r, th, ph):
    x = r * np.sin(th) * np.cos(ph)
    y = r * np.sin(th) * np.sin(ph)
    z = r * np.cos(th)
    u = np.array([x, y, z])
    return u

def db2lin(val, ref):
    return 10**(val/20.) * ref

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
    pls = pls / np.linalg.norm(pls)
    return pls

def legendreval(Ndec, thetas, phis, thetal, phil):
    '''Returns the value of the Legendre pulse at for a given steering direction'''
    nsrc = sph2cart(1., thetas, phis)
    nsteer = sph2cart(1., thetal, phil)
    costh = np.dot(nsrc, nsteer)
    val = 0
    for n in range(Ndec + 1):
        val += (2 * n + 1) / (4 * np.pi) * eval_legendre(n, costh)
    return val

# *********
def complexomp(Dct, y, n_nonzero_terms=10):
    res = y
    D = np.copy(Dct)
    ft = np.abs(np.matrix(D).T * res)
    inds = []
    ind = np.argmax(ft)
    inds.append(ind)
    Psi = np.matrix(D[:,ind]).T
    iter = 0
    ey = np.matrix(np.eye(len(y), len(y), dtype=complex))
    while iter < n_nonzero_terms:
        Pi = np.linalg.pinv(Psi)
        res = (ey - Psi * Pi) * y
        ft = np.abs(D.T * res)
        ind = np.argmax(ft)
        Psi = np.concatenate((Psi, np.matrix(D[:,ind]).T), 1)
        inds.append(ind)
        iter += 1
    alphas, _, _, _ = np.linalg.lstsq(Psi, y, rcond=None)
    # alphas = np.linalg.pinv(Psi) * np.matrix(y).T
    return inds, alphas, res

def nghalphas(doa, lev, gran, inds, alphas):
    indself = hp.ang2pix(2**lev, doa[0], doa[1], nest=True)
    indn = hp.get_all_neighbours(2**lev, doa[0], doa[1], nest=True) # Calculate the neighbours at the given resolution
    indall = list(indn)
    indall.append(indself)
    indset = set(indall)
    inda = []
    alpa = []
    for idx in range(len(inds)):
        idc = inds[idx]
        idc = idc // (4 ** (gran-lev))
        if idc in indset:
            alpa.append(alphas[idx])
            inda.append(idc)
    return inda, alpa

def maskedhalphas(doa, kappa, lev, gran, inds, alphas, offset):
    nsrd = sph2cart(1., doa[0], doa[1])
    inda = []
    alpa = []
    for idx in range(len(inds)):
        idc = inds[idx]
        idc = idc // (4 ** (gran - lev))
        th, ph = hp.pix2ang(2**lev, idc, nest=True)
        npix = sph2cart(1., th, ph)
        costheta = np.dot(nsrd, npix)
        gain = vonmises(kappa, costheta, offset) # Offset is hardcoded
        inda.append(idc)
        alpa.append(gain * alphas[idx])
    return inda, alpa


def vonmises(kappa, costheta, offset, normflag=True):
    I0 = spec.iv(0, kappa)
    if normflag:
        gain = np.exp(kappa * (costheta - 1))
    else:
        gain = np.exp(kappa * costheta) / (2 * np.pi * I0)
    gain = gain + offset
    return gain

######################## NEW! #############################

def vonmises2(dmu, deval, kappa, offset, normflag=True):
    I0 = spec.iv(0, kappa)
    nmu = sph2cart(1., dmu[0], dmu[1])
    neval = sph2cart(1., deval[0], deval[1])
    costheta = np.dot(nmu, neval)
    if normflag:
        gain = np.exp(kappa * (costheta - 1))
    else:
        gain = np.exp(kappa * costheta) / (2 * np.pi * I0)
    gain = gain + offset
    return gain

def suppressmask(deval, doasrc, kappasrc, doasupp, kappasupp, offset):
    assert kappasupp > kappasrc
    assert type(deval)==tuple
    # Calculate the beamform mask
    gain = vonmises2(doasrc, deval, kappasrc, 0.)
    for item in doasupp:
        gain *= (1-vonmises2(item, deval, kappasupp, 0.))
    gain += offset
    return gain

def maskedhalphas2(doasrc, doasupp, kappasrc, kappasupp, lev, gran, inds, alphas, offset):
    # nsrd = sph2cart(1., doa[0], doa[1])
    inda = []
    alpa = []
    for idx in range(len(inds)):
        idc = inds[idx]
        idc = idc // (4 ** (gran - lev))
        deval = hp.pix2ang(2**lev, idc, nest=True)
        # npix = sph2cart(1., th, ph)
        # costheta = np.dot(nsrd, npix)
        gain = suppressmask(deval, doasrc, kappasrc, doasupp, kappasupp, offset)
        inda.append(idc)
        alpa.append(gain * alphas[idx])
    return inda, alpa

######################## NEW! #############################

def recompose(inda, lev, alpa, doa, Nd):
    vl = 0j
    for idx in range(len(inda)):
        thl, phl = hp.pix2ang(2**lev, inda[idx], nest=True)
        vl += alpa[idx] * legendreval(Nd, doa[0], doa[1], thl, phl)
    return vl

# *********

# def complexomp_nzt(Dct, dt, nrm, y, n_nonzero_terms=10):
#     res = np.matrix(y).T
#     D = np.copy(Dct)
#     ft = np.abs(np.matrix(D).T * res)
#     inds = []
#     ind = np.argmax(ft)
#     inds.append(dt[ind])
#     nm = []
#     nm.append(nrm[ind])
#     Psi = np.matrix(D[:,ind]).T
#     # D[:, ind] = 0 # Would that be good to suppress the region as well?
#     iter = 1
#     ey = np.matrix(np.eye(len(y), len(y), dtype=complex))
#     while iter < n_nonzero_terms:
#         res = (ey - Psi * np.linalg.pinv(Psi))*np.matrix(y).T
#         ft = np.abs(D.T * res)
#         ind = np.argmax(ft)
#         inds.append(dt[ind])
#         nm.append(nrm[ind])
#         Psi = np.concatenate((Psi, np.matrix(D[:,ind]).T), 1)
#         iter += 1
#     alphas = np.diag(nm) * np.linalg.pinv(Psi) * np.matrix(y).T
#     return inds, alphas

def complexomp_doathr(Dct, y, Nd, lev, doas, thr, ep, maxiter=50):
    res = np.matrix(y).T
    Psi = np.matrix(np.zeros((12*2**(2*lev),0)), dtype=complex)
    D = np.copy(Dct)
    for item in doas:
        th = item[0]
        ph = item[1]
        pls = legendrepulse(Nd, lev, th, ph)
        res = res - (np.matrix(pls).T * np.conj(res).T) * np.matrix(pls).T
        Psi = np.concatenate((Psi, np.matrix(pls).T), 1)
    inds = []
    nrm = np.linalg.norm(res) # Norm of the residual
    eps = thr * nrm
    iter = 1
    while (nrm > eps) and (iter<=maxiter):
        iter += 1
        ft = np.abs(np.conj(D.T) * res)
        ind = np.argmax(ft)
        corrs = np.abs(np.array(Psi.T * np.matrix(D[:,ind]).T))**2
        if all(corrs <= (1-ep)):
            inds.append(ind)
            Psi = np.concatenate((Psi, np.matrix(D[:,ind]).T), 1)
            res = res - (np.matrix(D[:,ind]).T * np.conj(res).T) * np.matrix(D[:,ind]).T
            nrm = np.linalg.norm(res)
        else:
            D[:,ind] = 0
    Psinv = np.linalg.pinv(Psi)
    alphas = Psinv * np.matrix(y).T
    # alphas = alphas * 2 / (Nd + 1) / 1.1
    svecrec = Psi * alphas
    alphas = alphas / np.linalg.norm(svecrec) * np.linalg.norm(y)
    alphas = alphas * np.pi / (Nd + 1) / np.sqrt(3)
    return inds, alphas, Psinv

def complexomp_doathrw(Dct, y, Nd, lev, win, doas, thr, ep, maxiter=50):
    res = np.matrix(y).T
    Psi = np.matrix(np.zeros((12*2**(2*lev),0)), dtype=complex)
    D = np.copy(Dct)
    for item in doas:
        th = item[0]
        ph = item[1]
        pls = legendrepulse(Nd, lev, th, ph)
        res = res - (np.matrix(pls).T * np.conj(res).T) * np.matrix(pls).T
        Psi = np.concatenate((Psi, np.matrix(pls).T), 1)
    inds = []
    nrm = np.linalg.norm(res) # Norm of the residual
    eps = thr * nrm
    iter = 1
    while (nrm > eps) and (iter<=maxiter):
        iter += 1
        ft = np.abs(np.conj(D.T) * res)
        ind = np.argmax(ft)
        corrs = np.abs(np.array(Psi.T * np.matrix(D[:,ind]).T))**2
        if all(corrs <= (1-ep)):
            inds.append(ind)
            Psi = np.concatenate((Psi, np.matrix(D[:,ind]).T), 1)
            res = res - (np.matrix(D[:,ind]).T * np.conj(res).T) * np.matrix(D[:,ind]).T
            nrm = np.linalg.norm(res)
        else:
            D[:,ind] = 0

    alphas = np.linalg.pinv(Psi) * np.matrix(win * y).T
    # alphas = alphas * 2 / (Nd + 1) / 1.1

    # svecrec = Psi * alphas
    # alphas = alphas / np.linalg.norm(svecrec) * np.linalg.norm(y)

    alphas = alphas * np.pi / (Nd + 1) / np.sqrt(3)
    return inds, alphas

def complexomp_doa(Dct, y, n_nonzero_terms, Nd, lev, doas):
    res = np.matrix(y).T
    Psi = np.matrix(np.zeros((12*2**(2*lev),0)), dtype=complex)
    D = np.copy(Dct)
    # indd = range(12*2**(2*gran))
    for item in doas:
        th = item[0]
        ph = item[1]
        # vec = sph2cart(1., th, ph)
        # inds = hp.query_disc(2**gran, vec, radius= np.pi/(4*Nd), nest=True)
        # for jtem in inds:
        #     indd.remove(jtem)
        pls = legendrepulse(Nd, lev, th, ph)
        res = res - (np.matrix(pls).T * res.T) * np.matrix(pls).T
        Psi = np.concatenate((Psi, np.matrix(pls).T), 1)
    # D = D[:,indd]
    inds = []
    iter = 1
    while iter < n_nonzero_terms:
        ft = np.abs(np.conj(D.T) * res)
        ind = np.argmax(ft)
        inds.append(ind)
        Psi = np.concatenate((Psi, np.matrix(D[:,ind]).T), 1)
        res = res - (np.matrix(D[:,ind]).T * res.T) * np.matrix(D[:,ind]).T
        iter += 1
    # wd = np.ones(Psi.shape[0])
    # wd[0:len(doas)] = 0.2
    # W = np.matrix(np.diag(wd), dtype=complex)
    alphas = np.linalg.pinv(Psi) * np.matrix(y).T
    alphas = alphas * 2 / (Nd + 1) / 1.1
    return inds, alphas

def scomp(Dct, y, n_nonzero_terms=10, radius=np.pi/6): # Spatially constrained COMP with adaptive dictionary
    res = np.matrix(y).T
    D = np.copy(Dct)
    nside = hp.npix2nside(D.shape[1])

    ft = np.abs(np.conj(D.T) * res)
    inds = []
    ind = np.argmax(ft)
    th, ph = hp.pix2ang(nside, ind, nest=True)
    nv = sph2cart(1., th, ph)
    inds.append(ind)
    Psi = np.matrix(D[:,ind]).T

    ind0 = hp.query_disc(nside, nv, radius, nest=True)
    # st = (ind // 2**(2*delt)) * (2**(2*delt))
    # ed = st + (2**(2*delt))
    # ind0 = range(st, ed)
    D[:, ind0] = 0  # Would that be good to suppress the region as well?
    iter = 1
    ey = np.matrix(np.eye(len(y), len(y), dtype=complex))
    while iter < n_nonzero_terms:
        res = (ey - Psi * np.linalg.pinv(Psi))*np.matrix(y).T
        ft = np.abs(np.conj(D.T) * res)
        ind = np.argmax(ft)
        th, ph = hp.pix2ang(nside, ind, nest=True)
        nv = sph2cart(1., th, ph)
        inds.append(ind)
        ind0 = hp.query_disc(nside, nv, radius, nest=True)
        Psi = np.concatenate((Psi, np.matrix(D[:,ind]).T), 1)
        D[:, ind0] = 0
        iter += 1
    alphas = np.linalg.pinv(Psi) * np.matrix(y).T
    return inds, alphas

def complexomp_thr(Dct, y, thr):
    yn = np.copy(y)
    D = np.copy(Dct)
    res = np.matrix(yn).T
    nrm = np.linalg.norm(res)
    eps = nrm * thr
    ft = np.abs(np.conj(D.T) * res)
    inds = []
    ind = np.argmax(ft)
    inds.append(ind)
    Psi = np.matrix(D[:,ind]).T
    D[:, ind] = 0
    ey = np.matrix(np.eye(len(y), len(y), dtype=complex))
    while thr > eps:
        res = (ey - Psi * np.linalg.pinv(Psi))*np.matrix(y).T
        thr = np.linalg.norm(res)
        ft = np.abs(np.conj(D.T) * res)
        ind = np.argmax(ft)
        inds.append(ind)
        Psi = np.concatenate((Psi, np.matrix(D[:,ind]).T), 1)
    alphas = np.linalg.pinv(Psi) * np.matrix(y).T
    return inds, alphas

def selectdict(ths, phs, radius, lev, Dct):
    Dres = np.copy(Dct)
    nvec = sph2cart(1, ths, phs)
    inddisc = hp.query_disc(2**lev,  nvec, radius, nest=True)
    Ds = Dres[:, inddisc]
    Dres[:, inddisc] = 0
    return Ds, Dres

def planewave(freq, ra, A, thetas, phis, Nmax):
    mvec = []
    kra = 2 * np.pi * freq / 340. * ra
    jn, jnp, yn, ynp = spec.sph_jnyn(Nmax, kra)
    hn = jn - 1j * yn
    hnp = jnp - 1j * ynp
    bnkra = jn - (jnp / hnp) * hn
    b = []
    for n in range(Nmax + 1):
        b.append(bnkra[n])

    for n in range(Nmax + 1):
        for m in range(-n, n + 1):
            pnm = A * 4 * np.pi * (1j) ** n * b[n] * np.conj(spec.sph_harm(m, n, phis, thetas))
            mvec.append(pnm)
    return np.array(mvec), freq

# def compwithsourcedoas(srcdoas, radius, lev, Dct, y, n_nonzero_terms):
#     Dres = np.copy(Dct)
#     Psi = np.matrix(np.zeros((Dres.shape[0], len(srcdoas)), dtype=complex))
#     indi = 0
#     inds = []
#     for item in srcdoas:
#         ths = item[0]
#         phs = item[1]
#         Ds, Dres = selectdict(ths, phs, radius, lev, Dres)
#         res = np.matrix(y).T
#         ft = np.abs(np.conj(Ds.T) * res)
#         ind = np.argmax(ft)
#         inds.append(ind)
#         Psi[:,indi] = np.matrix(Ds[:, ind]).T
#         indi += 1
#     iter = 1
#     ey = np.matrix(np.eye(len(y), len(y), dtype=complex))
#     while iter < (n_nonzero_terms - len(srcdoas)):
#         res = (ey - Psi * np.linalg.pinv(Psi)) * np.matrix(y).T
#         ft = np.abs(np.conj(Dres.T) * res)
#         ind = np.argmax(ft)
#         inds.append(ind)
#         Psi = np.concatenate((Psi, np.matrix(Dres[:, ind]).T), 1)
#         iter += 1
#     alphas = np.linalg.pinv(Psi) * np.matrix(y).T
#     return inds, alphas

def selectnmax(freq, ra):
    k = 2 * np.pi * freq / 340.
    kra = k * ra
    if kra < 1:
        Nmax = 1
    elif kra < 2:
        Nmax = 2
    elif kra < 3:
        Nmax = 3
    else:
        Nmax = 4
    return Nmax



if __name__ =='__main__':
    lev = 1
    gran = 4
    freq = 2000.
    ra = 4.2e-2
    Nmax = selectnmax(freq, ra)
    print(Nmax)
    # pls = legendrepulse(4, lev, np.pi/2, np.pi / 3)
    # pls = pls * (0.4 - .3j)
    # pls2 = legendrepulse(4, lev, np.pi / 2, -np.pi / 3)
    # pls2 = pls2 * (-0.9 + .7j)
    # pls += pls2
    # pls3 = legendrepulse(4, lev, np.pi / 4, 0)
    # pls3 = pls3 * (0.5 + .5j)
    # pls += pls3
    D = legendredict(Nmax, lev, gran)
    mvec1, freq1 = planewave(freq, 4.2e-2,  0.3 - 0.3j, np.pi / 2,  0.,       Nmax)
    mvec2, freq2 = planewave(freq, 4.2e-2, -0.9 - 0.2j, np.pi / 2., np.pi / 2., Nmax)
    mvec3, freq3 = planewave(freq, 4.2e-2, 0.15 - 0.15j, np.pi / 2., 3 * np.pi / 2., Nmax)
    ydict = getYnmtlookup('/Users/huseyinhacihabiboglu/PycharmProjects/Separate/ynmdict/ynm_N'+ str(Nmax) +'_lvl' + str(lev) + '.pkl')
    svec, Nd = srford(mvec1 + mvec2, freq, 4.2e-2, lev, ydict)
    doas = [(np.pi/2,  0.), (np.pi/2., np.pi/2.), (np.pi / 2., 3 * np.pi / 2.)]
    start = tm.time()
    for id in range(1000):
        inds, alphas, Psi = complexomp_doathr(D, svec, Nmax, lev, doas, thr=0.01, ep=0.2, maxiter=10)
        # svecrec = Psi * alphas
        # plt.plot(abs(svecrec))
        # plt.plot(abs(svec))
        # plt.show()
        # # yN = beamform(D, inds, alphas)
        # # print(yN)
        # # print(svec)
        # # print(alphas)
    stop = tm.time()

    print(stop-start)

    # inds, alphas = complexomp_thr(D, pls)
    doa = np.zeros(3072, dtype=complex)
    doa[inds]=np.array(alphas[3:]).reshape((len(alphas[3:]),))
    hp.mollview(np.real(doa), nest=True)
    plt.show()