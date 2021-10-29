import numpy as np
from operator import itemgetter
from itertools import groupby


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
