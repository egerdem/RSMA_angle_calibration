import scipy.special as sp
import numpy as np
from collections import defaultdict
import pandas as pd
from itertools import chain

def cart2sph(x, y, z):
    """
    r, th, ph = cart2sph(x, y, z)

    Return the spherical coordinate representation of point(s) given in
    Cartesian coordinates

    r: radius, th: elevation angle defined from the positive z axis,
    ph: the azimuth angle defined from the positive x axis
    """
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    th = np.arccos(z / r)
    ph = np.arctan2(y, x)
    return r, th, ph

def key_diff(dict1, dict2):
    keys1 = dict1.keys()
    keys2 = dict2.keys()
    key = keys1 - keys2
    value = np.zeros(len(key)) 
    return(dict(zip(key,value)))  

def merge_two_dicts(x, y):  
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None    
    return z

def bnm(n, m):
    if (0 <= m <= n):
        return np.sqrt((n-m-1)*(n-m)/((2*n-1)*(2*n+1)))
    if (-n <= m < 0):
        return -np.sqrt((n - m - 1) * (n - m) / ((2 * n - 1) * (2 * n + 1)))
    elif abs(m)>n:
        return 0

def anm(n, m):
    if (abs(m) <= n):
        return np.sqrt((n+abs(m)+1)*(n+1-abs(m))/((2*n+1)*(2*n+3)))
    else:
        return 0
    
def mic_loc(x,y,z):
#    origin = [0,0,0]
    return np.array([x, y, z])

def mic_sub(mic_p, mic_q):
    mic_pq = mic_p - mic_q
    return mic_pq

def Snm(n, m, k, rpq): #Gum2002 - Eq10
    return(sp.spherical_jn(n, k * rpq[0])+1j * sp.spherical_yn(n, k * rpq[0])) * sph_harm(m, n, rpq[2], rpq[1])    

def sph_harm(m, n, phi, theta):
    sph = (-1)**m * np.sqrt((2*n+1)/(4*np.pi) * sp.factorial(n-np.abs(m)) / sp.factorial(n+np.abs(m))) * sp.lpmn(int(np.abs(m)), int(n), np.cos(theta))[0][-1][-1] * np.exp(1j*m*phi)
    return sph

def Snm_direct(n,m,k,rp_sph):
    jn = sp.spherical_jn(n, k * rp_sph[0])
    yn = sp.spherical_yn(n, k * rp_sph[0])
    hn = jn + 1j * yn
    Ynm = sph_harm(m, n, rp_sph[2], rp_sph[1])
    Snm = hn * Ynm
    return Snm

def dict2matrix(d,l_max, n_max):
    #takes dict key index to compute 2-D matrix represantation
    rt_matrix = np.zeros(((l_max+1)**2,(n_max+1)**2))
    rt_matrix = np.zeros(((l_max+1)**2,(n_max+1)**2),dtype=np.complex)
    for key in d.keys():
        l,s,n,m = key[0],key[1],key[2],key[3]                    
        r = (l+1)**2 - (l-s)
        t = (n+1)**2 - (n-m)
        if d[key] != 0:
            rt_matrix[r-1][t-1] = 11
            rt_matrix[r-1][t-1] = d[key]
    return rt_matrix

def dict2matrix_ref(d,l_max, n_max):
    #compute 2-D matrix from a dictionary, where elements as keys strings
    rt_matrix = np.zeros(((l_max+1)**2,(n_max+1)**2))
    rt_matrix = np.zeros(((l_max+1)**2,(n_max+1)**2), dtype='object')
    for key in d.keys():
        l,s,n,m = key[0],key[1],key[2],key[3]                    
        r = (l+1)**2 - (l-s)
        t = (n+1)**2 - (n-m)
        tt = ''.join([str(x) for x in key])
        rt_matrix[r-1][t-1] = tt
        df = pd.DataFrame(rt_matrix)
    return df

def setsectorials(Lmax, Nmax):
    sectorials = defaultdict()
    for l in range(Lmax + 1):
        for s in range(-l, l + 1):
            for m in range(min(-1,-Nmax), Nmax + 1):
                sectorials[(l, s, np.abs(m), m)] = 0
    return sectorials

def set_symsectorials(Lmax, Nmax):
    symsectorials = defaultdict()
    for l in range(Lmax + 1):
        for s in range(-l, l + 1):
            for m in range(min(-1,-Nmax), Nmax + 1):
                symsectorials[(np.abs(m), -m, l, -s)] = 0
    return symsectorials

def allcoefs(Lmax, Nmax):
    sectorials = defaultdict()
    for l in range(Lmax + 1):
        for n in range(Nmax + 1):
            for s in range(-l, l + 1):
                for m in range(-n, n + 1):
                    sectorials[(l, s, n, m)] = 0
    return(sectorials)
     
def symmetric_SRsectorial(d):
    symsectorials = defaultdict()
    for key in d.keys():
        newkey = (np.abs(key[2]), -key[3], key[0], -key[1])
        if newkey not in d.keys():            
            symsectorials[newkey] = d[key]*(-1.0)**(key[0]+key[3])
    return symsectorials

def SRsectorial(l, s, n, m, k, rpq):
#    orhun-ege, sadece indexleri değiştirmek
    if issect((l,s,n,m)):
        if (l < 0) | (n < 0):
#            print("ln negative,not exist, zero", l,s,n,m)
            return 0
        
        elif (np.abs(s)>l) | (np.abs(m)>n):
#            print("not exist, zero", l,s,n,m)
            return 0
        
        elif n==m==0:
#            print("nm 00 used",l,s,n,m)
            return(((-1)**l)*np.sqrt(4 * np.pi)*Snm(l, -s, k, rpq))
            
        elif l==s==0:
#            print("ls 00 used",l,s,n,m)
            return (np.sqrt(4*np.pi) * Snm(n, m, k, rpq))
        
        elif m>=0:
#            print("m>0 used",l,s,n,m,"these needed:",l - 1, s - 1, m-1, m-1,"     ",l+1, s-1, m-1, m-1)
            return (bnm(l, -s) * SRsectorial(l - 1, s - 1, m-1, m-1, k, rpq) - bnm(l+1, s-1) * SRsectorial(l+1, s-1, m-1, m-1, k, rpq)) / bnm(m, -m)
        
        elif m<=0:
#            print("m<0 used",l,s,n,m,"these needed:",l - 1, s + 1, np.abs(m+1), m+1,"     ",l+1, s+1, -(m+1), m+1)
            return (bnm(l, s) * SRsectorial(l - 1, s + 1, np.abs(m+1), m+1, k, rpq) - bnm(l+1, -s-1) * SRsectorial(l+1, s+1, -(m+1), m+1, k, rpq)) / bnm(-m, m)
        else:
            print("line188")
            return np.nan
    else:
        print("this key is not even a sectorial key!!!")
        
def del_keylist(dic, keys):        
    #delete keys from a dic
    for key in keys:
        del dic[keys]
    return(dic)
    
def key2list(dic):
    t = list(dic.keys())
    res = [list(row) for row in t]
    return(res)

def list2key(key):
    #takes lists returns dics
    value = np.zeros(len(key))
    return(dict(zip(key,value)))
    
def keyinfo_nm(key):    
    # which non-sectorial SR coeff we can calculate from a given key nm sectorial
    l,s,n,m = key[0],key[1],key[2],key[3]
    a,b,c = (l+1,s,n,m), (l,s,n-1,m), (l-1,s,n,m)
    print("abc:", a, b, c)
    res = (l, s, n+1, m)
    print("res:",res) #resulting key
    return()

def structure_nm(dic):   
    keys_list = key2list(dic)
    keys_reached = []
    while(len(keys_list)>2):    
        keys_list = keys_list[1:-1]
        new_keys = []
        for i in range(len(keys_list)):        
            origin = keys_list[i]
            l,s,n,m = origin
            res = (l, s, n+1, m) 
            new_keys.append(res)
        keys_reached.append(new_keys)
        keys_list = new_keys
    return(keys_reached)
   
def structure_ls(dic):   
    keys_list = key2list(dic)
    keys_reached = []
    while(len(keys_list)>2):    
        keys_list = keys_list[1:-1]
        new_keys = []
        for i in range(len(keys_list)):        
            origin = keys_list[i]
            l,s,n,m = origin
            res = (l+1, s, n, m) 
            new_keys.append(res)
        keys_reached.append(new_keys)
        keys_list = new_keys
    return(keys_reached)

def unpack(list_):
    return(list(chain.from_iterable(list_)))

def issect(key):
    l,s,n,m = key[0],key[1],key[2],key[3]                    
    if l == abs(s) or n == abs(m):
        return(True)
    else: return(False)
    
def rec_nm(l,s,n,m,pool):
    b = (l, s, n-1, m)
    a = (l+1, s, n, m) 
    c = (l - 1, s, n, m)
    target = (l, s, n+1, m)
    if issect(target):
        print("target is already a sect, returned from allsect")
        return(allsect[target])
    else:        
        if a[2]<np.abs(a[3]) or a[0]<np.abs(a[1]): #(n<np.abs(m) or l<np.abs(s))
#            print("a is zero, %d %d %d %d = 0 added to dict" %(a[0],a[1],a[2],a[3]))
            pool[a[0],a[1],a[2],a[3]] = 0
        if b[2]<np.abs(b[3]) or b[0]<np.abs(b[1]): #(n<np.abs(m) or l<np.abs(s))
#            print("b is zero, %d %d %d %d = 0 added to dict" %(b[0],b[1],b[2],b[3]))
            pool[b[0],b[1],b[2],b[3]] = 0
        if c[2]<np.abs(c[3]) or c[0]<np.abs(c[1]): #(n<np.abs(m) or l<np.abs(s))
#            print("c is zero, %d %d %d %d = 0 added to dict" %(c[0],c[1],c[2],c[3]))        
            pool[c[0],c[1],c[2],c[3]] = 0    
            
        result = (anm(n-1, m) * pool[l, s, n-1, m] - \
                     anm(l, s) * pool[l+1, s, n, m] + \
                     anm(l-1, s) * pool[l - 1, s, n, m])/anm(n, m) 
#        print("result:",target,result)
        return(result)
   
def rec_ls(l,s,n,m,pool):
    b = (l - 1, s, n, m)
    a = (l, s, n+1, m)
    c = (l, s, n-1, m)
    target = (l+1, s, n, m)
#    print("origin:",key, "abc_ls:", a, b, c)
    if issect(target):
        print("target is already a sect, returned from allsect")

        return(allsect[target])
    else:        
        if a[2]<np.abs(a[3]) or a[0]<np.abs(a[1]): #(n<np.abs(m) or l<np.abs(s))
#            print("a is zero, %d %d %d %d = 0 added to dict" %(a[0],a[1],a[2],a[3]))
            pool[a[0],a[1],a[2],a[3]] = 0
        if b[2]<np.abs(b[3]) or b[0]<np.abs(b[1]): #(n<np.abs(m) or l<np.abs(s))
#            print("b is zero, %d %d %d %d = 0 added to dict" %(b[0],b[1],b[2],b[3]))
            pool[b[0],b[1],b[2],b[3]] = 0
        if c[2]<np.abs(c[3]) or c[0]<np.abs(c[1]): #(n<np.abs(m) or l<np.abs(s))
#            print("c is zero, %d %d %d %d = 0 added to dict" %(c[0],c[1],c[2],c[3]))        
            pool[c[0],c[1],c[2],c[3]] = 0    
        
        result = (anm(l-1, s) * pool[l - 1, s, n, m] + \
                     anm(n-1, m) * pool[l, s, n-1, m] - \
                     anm(n,m) * pool[l, s, n+1, m])/anm(l,s)
#        print("result:",target,result)
        return(result) 

def traverse_nm_A(keys_list,pool):    
    while(len(keys_list)>1):    
#        print("somet")
        keys_list = keys_list[1:-1]
        new_keys = []        
        for i in range(len(keys_list)):     
#            print("i",i)
            origin = keys_list[i]
            l,s,n,m = origin
            target = (l, s, n+1, m) 
            if not (target[0]>Lmax or target[1]>Lmax or target[2]>Lmax or target[3]>Lmax):
                new_keys.append(target)
#                print("target_nm:", target)
                bos[target] = 111111
                pool[target] = rec_nm(l, s, n, m, pool) 
                all_coeffs[target] = rec_nm(l, s, n, m, pool) 
        keys_list = new_keys
    return

def traverse_ls_A(keys_list,pool):
    while(len(keys_list)>2):    
        keys_list = keys_list[1:-1]
        new_keyss = []
        for i in range(len(keys_list)):        
            origin = keys_list[i]
            l,s,n,m = origin
            target = (l+1, s, n, m) 
            if not (target[0]>Lmax or target[1]>Lmax or target[2]>Lmax or target[3]>Lmax):

                new_keyss.append(target)
#                print("target_ls:", target)
                bos[target] = 5555555
                pool[target] = rec_ls(l, s, n, m, pool) 
                all_coeffs[target] = rec_ls(l, s, n, m,pool) 
        keys_list = new_keyss
    return

def sm_sectorials(Lmax, Nmax):
    smm = defaultdict()
    for s in range(-Lmax, Lmax + 1):
        for m in range(min(-1,-Nmax), Nmax + 1):
            smm[(s,m)] = []       
    return smm

def lsmn_2_rt(l,s,n,m):
    r = (l+1)**2 - (l-s)
    t = (n+1)**2 - (n-m)
    return(r,t)

def needed_lmax(s,m,Nt):
    Lmax=2*Nt-max(abs(s),abs(m))
    return(Lmax)
        
def max_order(Nt, m_try):
    maxx = [] 
    for l in range(0, Nt + 1):
        for s in range(-l, l + 1):
            print(s,m_try,Nt,needed_lmax(s,m_try,Nt))
            maxx.append(needed_lmax(s,m_try,Nt)+m_try)  
    dum = np.max(maxx)
    return(dum)

def reexp_coef(k, Lmax, Nmax, rpq_sph):
    N = (Lmax - 1) ** 2
    #rpq_sph = cart2sph(rpq[0], rpq[1], rpq[2])

    sect = setsectorials(Lmax, Nmax)
    bos = defaultdict()

    """  SECTORIALS  """

    for item in sect.keys():
        sect[item] = SRsectorial(item[0], item[1], item[2], item[3], k, rpq_sph)

    """  SYMMETRIC SECTORIALS  """

    sym_sectKEYS = set_symsectorials(Lmax, Nmax)  # only the keys of symmetric sectorials
    sym_sect = symmetric_SRsectorial(sect)

    allsect = merge_two_dicts(sym_sect, sect)  # all sectorials merged
    allsect_pool = merge_two_dicts(sym_sect, sect)  # all sectorials merged

    """ TESSERALS  """

    all_coeffs = defaultdict()

    for key in allsect.keys():
        all_coeffs[key] = allsect[key]

    smm = sm_sectorials(Lmax, Nmax)
    #   d = set()
    # for key in sym_sectKEYS.keys():
    #    l,s,n,m = key[0],key[1],key[2],key[3]
    #    d = d.union(set([(s,m)]))
    nm_sect = sm_sectorials(Lmax, Nmax)
    ls_sect = sm_sectorials(Lmax, Nmax)

    for key in sect.keys():
        l, s, n, m = key[0], key[1], key[2], key[3]
        if issect(key):
            nm_sect[(s, m)].append(key)

    for key in sym_sectKEYS.keys():
        l, s, n, m = key[0], key[1], key[2], key[3]
        if issect(key):
            ls_sect[(s, m)].append(key)

    rt_matrix_all = dict2matrix(all_coeffs, Lmax, Nmax)
    return rt_matrix_all[:N, :N]


if __name__=='__main__':
       
    m_try = 2
    n_try = 5
    Lmax, Nmax = 7,7
    k = 1
    rp = mic_loc(1, -6, 1)
    rp_sph = cart2sph(rp[0], rp[1], rp[2])

    rq = mic_loc(-1, 1, 0)
    rq_sph = cart2sph(rq[0], rq[1], rq[2])

    rpq = mic_sub(rp, rq)    
    rpq_sph = cart2sph(rpq[0], rpq[1], rpq[2])
    
    sect = setsectorials(Lmax, Nmax)
    bos = defaultdict()
    
    """  SECTORIALS  """
    
    for item in sect.keys():
        sect[item] = SRsectorial(item[0], item[1], item[2], item[3], k, rpq_sph)       
    
    """  SYMMETRIC SECTORIALS  """

    sym_sectKEYS = set_symsectorials(Lmax, Nmax)    # only the keys of symmetric sectorials
    sym_sect = symmetric_SRsectorial(sect)
    
    allsect = merge_two_dicts(sym_sect, sect)       # all sectorials merged
    allsect_pool = merge_two_dicts(sym_sect, sect)  # all sectorials merged
    
    """ TESSERALS  """
    
    all_coeffs = defaultdict()
    
    for key in allsect.keys():    
        all_coeffs[key] = allsect[key]       
    
    smm = sm_sectorials(Lmax, Nmax)
    #   d = set()
    #for key in sym_sectKEYS.keys():
    #    l,s,n,m = key[0],key[1],key[2],key[3]                   
    #    d = d.union(set([(s,m)]))
    nm_sect = sm_sectorials(Lmax, Nmax)
    ls_sect = sm_sectorials(Lmax, Nmax)
    
    for key in sect.keys():
        l,s,n,m = key[0],key[1],key[2],key[3]       
        if issect(key):
            nm_sect[(s,m)].append(key)
            
    for key in sym_sectKEYS.keys():
        l,s,n,m = key[0],key[1],key[2],key[3]       
        if issect(key):
            ls_sect[(s,m)].append(key)
                   
    bos = defaultdict()            
    for key in nm_sect.keys():
        sublist = nm_sect[key]  
        traverse_nm_A(sublist, allsect_pool)   
    
    for key in ls_sect.keys():
        sublistt = ls_sect[key]  
        traverse_ls_A(sublistt, allsect_pool)           
    
    for i in range(5):
        Nt = i
        Sum = 0        
        for l in range(0, Nt + 1):
            jn = sp.spherical_jn(l, k * rq_sph[0])
            for s in range(-l, l + 1):
                Ynm = sph_harm(s, l, rq_sph[2], rq_sph[1])
                Rls = jn * Ynm
                key = (l,s,n_try,m_try)
                Sum += all_coeffs[key]*Rls               
#                print("for key:", key)            
#                print("n from", abs(m_try),2*Nt-max(abs(s),abs(m_try)))            
#                print("l from", abs(s),2*Nt-max(abs(s),abs(m_try)))
                    
                if all_coeffs[(l,s,n_try,m_try)]==0:
                    print("be careful! coeff is 0!")
        
        S_direct = Snm_direct(n_try, m_try, k, rp_sph)
        print("S_dir", S_direct)        
        print("S_sum:", Sum,"Nt:", Nt , "\n")

    #VERIFY
    m = 3
    l, s = 4, 2
    key
    left = bnm(m+1, -m-1) * all_coeffs[(l, s, m+1, m+1)]
    right = bnm(l, -s) * all_coeffs[(l-1, s-1, m, m)] - bnm(l+1, s-1) * all_coeffs[(l+1, s-1, m, m)]
    res = left - right
    print(res)

    print(all_coeffs[(2, 0 ,3, 1)])
    print(all_coeffs[(3, -1, 2, 0)])