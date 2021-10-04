import numpy as np
import pylab as plt
import batman, pdb
from tls_object import transit_search


def inject_custom_planet(ts, P, RpRearth, incdeg=90, ecc=0):
    '''
    Inject one planet with user-defined parameters into an existing light
    curve contained within a tls object.
    '''
    # get transit model
    t0 = float(np.random.uniform(ts.lc.bjd.min(), ts.lc.bjd.max()))
    omegadeg = float(np.random.uniform(0,360))
    args = ts.lc.bjd, ts.star.Ms, ts.star.Rs, P, t0, RpRearth, incdeg, ecc, omegadeg, ts.star.ab
    ts.lc.injected_model = transit_model(*args)

    # inject transiting planet
    ts.lc.fdetrend *= ts.lc.injected_model



def transit_model(bjd, Ms, Rs, P, T0, RpRearth, incdeg, ecc, omegadeg, ab):
    aRs = rvs.AU2m(rvs.semimajoraxis(P, Ms, 0)) / rvs.Rsun2m(Rs)
    rpRs = rvs.Rearth2m(RpRearth) / rvs.Rsun2m(Rs)
    params_dict = {'per': P, 't0': T0, 'rp': rpRs, 'a': aRs, 'inc': incdeg,
                   'ecc': ecc, 'w': omegadeg, 'u': list(ab)}

    params = batman.TransitParams()
    params.limb_dark = 'quadratic'
    for k in params_dict.keys():
        setattr(params, k, params_dict[k])

    # compute transit model
    m = batman.TransitModel(params, bjd)
    return m.light_curve(params)
