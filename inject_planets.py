import numpy as np
import pylab as plt
import batman, misc, pdb
from tls_object import transit_search


def inject_custom_planet(ts, P, RpRearth, b=0, ecc=0):
    '''
    Inject one planet with user-defined parameters into an existing light
    curve contained within a tls object.
    '''
    # get transit model
    t0 = float(np.random.uniform(ts.lc.bjd.min(), ts.lc.bjd.max()))
    omegadeg = float(np.random.uniform(0,360))
    args = ts.lc.bjd, ts.star.Ms, ts.star.Rs, P, t0, RpRearth, b, ecc, omegadeg, ts.star.ab
    ts.lc.injected_model = transit_model(*args)

    # inject transiting planet
    ts.lc.fdetrend *= ts.lc.injected_model



def transit_model(bjd, Ms, Rs, P, T0, RpRearth, b, ecc, omegadeg, ab):
    aRs = misc.AU2m(misc.semimajoraxis(P, Ms, 0)) / misc.Rsun2m(Rs)
    rpRs = misc.Rearth2m(RpRearth) / misc.Rsun2m(Rs)
    inc = float(misc.inclination_aRs(aRs, b, ecc=ecc, omega_deg=omegadeg))
    params_dict = {'per': P, 't0': T0, 'rp': rpRs, 'a': aRs, 'inc': inc,
                   'ecc': ecc, 'w': omegadeg, 'u': list(ab)}

    params = batman.TransitParams()
    params.limb_dark = 'quadratic'
    for k in params_dict.keys():
        setattr(params, k, params_dict[k])

    # compute transit model
    m = batman.TransitModel(params, bjd)
    return m.light_curve(params)
