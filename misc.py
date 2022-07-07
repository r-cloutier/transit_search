import numpy as np
from operator import itemgetter
from itertools import groupby
import pdb

global G, Msun, Mearth, Rsun, Rearth, AU, pc, kb, mproton
G, Msun, Mearth = 6.67e-11, 1.98849925145e30, 6.04589804468e24
Rsun, Rearth, AU, pc = 695500e3, 6371e3, 1.495978707e11, 3.08567758149137e16
kb, mproton = 1.38e-23, 1.67e-27


def Rearth2m(r):
    return r*Rearth
def m2Rearth(r):
    return r/Rearth
def Rsun2m(r):
    return r*Rsun
def m2Rsun(r):
    return r/Rsun
def AU2m(r):
    return r*AU
def m2AU(r):
    return r/AU
def days2sec(t):
    return t*24.*60*60
def sec2days(t):
    return t/(24.*60*60)
def Msun2kg(m):
    return m*Msun
def kg2Msun(m):
    return m/Msun
def Mearth2kg(m):
    return m*Mearth
def kg2Mearth(m):
    return m/Mearth


def inclination_aRs(a_Rs, b, ecc=0., omega_deg=0.):
    '''Compute the inclination from the impact parameter and a/Rstar.'''
    omega = np.deg2rad(omega_deg)
    return np.rad2deg(np.arccos(b / a_Rs * ((1+ecc*np.sin(omega)) / (1-ecc**2))))



def semimajoraxis(P_days, Ms_Msun, mp_Mearth):
    ''' Compute the semimajor axis in AU from Kepler's third law.'''
    P, Ms, mp = days2sec(P_days), Msun2kg(Ms_Msun), Mearth2kg(mp_Mearth)
    return m2AU((G*(Ms+mp)*P*P / (4*np.pi*np.pi)) **(1./3))



def get_consecutive_sectors(sectors):
    '''
    Returns a list of lists where each element is a list of consecutive sectors.
    
    E.g. for a target observed in sectors 1,2,3,13,14,20, this function will return

    [[1, 2, 3], [13, 14], [20]]
    '''
    ranges=[]
    for k,g in groupby(enumerate(sectors),lambda x:x[0]-x[1]):
        group = (map(itemgetter(1),g))
        group = list(map(int,group))
        ranges.append(group if len(group) == 1 else list(range(group[0],group[-1]+1)))
    return ranges



def foldAt(t, P, T0=0):
    phi = ((t-T0) % P) / P
    phi[phi>.5] -= 1   # change range from [0,1] to [-0.5,0.5]
    return phi


def estimate_snr_deprecated(ts, P, Rp):
    Z = (Rearth2m(Rp) / Rsun2m(ts.star.Rs))**2
    sig = np.median(ts.lc.efnorm_rescaled)
    #sig = np.median(abs(ts.lc.fdetrend - np.median(ts.lc.fdetrend)))
    dT = 27 * np.max([len(s) for s in ts.lc.sect_ranges])  # max consecutive sectors
    return Z/sig * np.sqrt(dT/P)


def estimate_snr(ts, P, T0, Rp):
    Z = (Rearth2m(Rp) / Rsun2m(ts.star.Rs))**2
    sig = np.nanmedian(ts.lc.efnorm_rescaled)
    # get max number of transits per sect_range because I'm not searching all available sectors at once (only consecutive sectors)
    Ntransits = np.zeros(len(ts.lc.sect_ranges))
    for i,s in enumerate(ts.lc.sect_ranges):
        g = np.in1d(ts.lc.sectors, s)
        transit_times = [T0+n*P for n in np.arange(np.floor((ts.lc.bjd[g].min()-T0)/P), np.ceil((ts.lc.bjd[g].max()-T0)/P)+1)]
        Ntransits[i] = np.sum([np.any(np.isclose(ts.lc.bjd[g]-t, 0, atol=2/(60*24))) for t in transit_times])
    return Z/sig * np.sqrt(np.max(Ntransits))


def estimate_Prot_snr(ts, amp_ppt, Prot):
    T = 27 * np.unique(ts.lc.sectors).size
    return amp_ppt*1e-3 / np.median(ts.lc.efnorm_rescaled) * np.sqrt(T/Prot)


def compute_instellation(Teff, Rs, Ms, P):
    L = Rs**2 * (Teff/5780)**4
    F = L / semimajoraxis(P, Ms, 0)**2
    return F


def sinemodel(x, A, t0, P, offset):
    return A * np.sin(2*np.pi*(x-t0)/P) + offset


def lnlike(y, ey, model):
    assert np.all(np.isfinite(np.ascontiguousarray(ey)))
    return -.5 * np.nansum((y-model)**2 / ey**2)


def DeltaBIC(y, ey, model, modelnull, k=5, knull=1):
    '''Transit model (from the TLS) vs the null model (i.e. a flat line)'''
    BIC_model = k*np.log(y.size) - 2*lnlike(y, ey, model)   # theta = {P,T0,D,Z,baseline}
    BIC_null = knull*np.log(y.size) - 2*lnlike(y, ey, modelnull)
    return BIC_model, BIC_null, BIC_model - BIC_null


def bin_lc(x_fold, y, bin_width_min=60):
    bins = np.arange(x_fold.min(), x_fold.max(), bin_width_min/1440)
    denom, _ = np.histogram(x_fold, bins)
    num, _ = np.histogram(x_fold, bins, weights=y)
    denom[num == 0] = 1.0
    xbin = 0.5 * (bins[1:] + bins[:-1])
    ybin = num / denom
    return xbin, ybin
