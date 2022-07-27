import numpy as np
import pylab as plt
import exoplanet as xo
import pymc3 as pm
import aesara_theano_fallback.tensor as tt
import pymc3_ext as pmx
from celerite2.theano import terms, GaussianProcess
from gls import Gls
import pickle, pdb, misc
import astropy.units as u


MAD = lambda x: np.median(abs(x-np.median(x)))


def build_model_0planets_Rotation(x, y, ey, Prot, mask=None, start=None):

    if mask is None:
        mask = np.ones(len(x), dtype=bool)

    with pm.Model() as model:

        # Parameters for the stellar properties
        mean = pm.Normal("mean", mu=0.0, sd=10.0)

        # Transit jitter & GP parameters
        #sigma_rot = pm.InverseGamma(
        #    "sigma_rot", **pmx.estimate_inverse_gamma_parameters(1.0, 5.0)
        #)
        log_sigma_rot = pm.Uniform("log_sigma_rot", lower=-3, upper=np.log(np.std(y[mask])))
        sigma_rot = pm.Deterministic("sigma_rot", tt.exp(log_sigma_rot))
        log_period = pm.Normal("log_period", mu=np.log(Prot), sigma=.1)
        period = pm.Deterministic("period", tt.exp(log_period))
        log_Q0 = pm.HalfNormal("log_Q0", sigma=2)
        log_dQ = pm.Normal("log_dQ", mu=0, sigma=2)
        f = pm.Uniform("f", lower=0.1, upper=1.0)
        log_sigma_lc = pm.Normal("log_sigma_lc", mu=np.log(np.median(ey[mask])), sd=.1)

        # Set up the Gaussian Process model
        ##kernel = terms.SHOTerm(sigma=sigma, rho=rho, Q=1 / 3.0)
        kernel = terms.RotationTerm(
            sigma=sigma_rot,
            period=period,
            Q0=tt.exp(log_Q0),
            dQ=tt.exp(log_dQ),
            f=f,
        )
        gp = GaussianProcess(kernel, t=x[mask], yerr=tt.exp(log_sigma_lc))
        resid = y[mask]
        gp.marginal("gp", observed=resid)

        # Fit for the maximum a posteriori parameters, I've found that I can get
        # a better solution by trying different combinations of parameters in turn
        if start is None:
            start = model.test_point
        map_soln = pmx.optimize(start=start, vars=[log_sigma_lc, log_sigma_rot, log_Q0, log_dQ, f])
        map_soln = pmx.optimize(start=map_soln, vars=[mean])
        map_soln = pmx.optimize(start=map_soln, vars=[log_sigma_lc, log_sigma_rot, log_Q0, log_dQ, f])
        map_soln = pmx.optimize(start=map_soln)

        extras = dict(zip(["gp_pred"],
                          pmx.eval_in_model([gp.predict(resid)], map_soln),))

    return model, map_soln, extras


def build_model_0planets_SHO(x, y, ey, Prot, mask=None, start=None):

    if mask is None:
        mask = np.ones(len(x), dtype=bool)

    with pm.Model() as model:

        # Parameters for the stellar properties
        mean = pm.Normal("mean", mu=0.0, sd=10.0)

        # Transit jitter & GP parameters
        log_sigma_gp = pm.Uniform("log_sigma_gp", lower=-3, upper=np.log(np.std(y[mask])))
        log_rho_gp = pm.Normal("log_rho_gp", mu=np.log(Prot), sd=.2)
        log_tau_gp = pm.Uniform("log_tau_gp", lower=np.log(10*Prot), upper=20)
        ##log_sigma_lc = pm.Normal("log_sigma_lc", mu=np.log(MAD(y[mask])), sd=.5)
        log_sigma_lc = pm.Normal("log_sigma_lc", mu=np.log(np.median(ey[mask])), sd=.1)

        # GP model for the light curve
        kernel = terms.SHOTerm(
            sigma=tt.exp(log_sigma_gp),
            rho=tt.exp(log_rho_gp),
            tau=tt.exp(log_tau_gp),
        )
        gp = GaussianProcess(kernel, t=x[mask], yerr=tt.exp(log_sigma_lc))
        resid = y[mask]
        gp.marginal("gp", observed=resid)

        # Fit for the maximum a posteriori parameters, I've found that I can get
        # a better solution by trying different combinations of parameters in turn
        if start is None:
            start = model.test_point
        map_soln = pmx.optimize(start=start,
                                vars=[log_sigma_lc, log_sigma_gp, log_rho_gp,
                                      log_tau_gp])
        map_soln = pmx.optimize(start=map_soln, vars=[mean])
        map_soln = pmx.optimize(start=map_soln,
                                vars=[log_sigma_lc, log_sigma_gp, log_rho_gp,
                                      log_tau_gp])
        map_soln = pmx.optimize(start=map_soln)

        extras = dict(zip(["gp_pred"],
                          pmx.eval_in_model([gp.predict(resid)], map_soln),))

    return model, map_soln, extras



def _sigma_clip(fT, extras, sig=5):
    resid = fT - extras["gp_pred"]
    mask = abs(resid - np.nanmedian(resid)) < sig*MAD(resid)
    return mask



def convert2exoplanet(bjd, fnorm, efnorm):
    ref_time = np.nanmean([np.nanmax(bjd), np.nanmin(bjd)])
    return bjd-ref_time, 1e3*(fnorm-1), 1e3*efnorm, ref_time



def convert2norm(bjd_shift, fexo, efexo, ref_time):
    return bjd_shift+ref_time, 1e-3*fexo+1, 1e-3*efexo




def detrend_light_curve(bjd, fnorm, efnorm, sectors, sect_ranges, Prot):
    assert bjd.size == fnorm.size
    assert bjd.size == efnorm.size
    assert bjd.size == sectors.size

    fdetrend, mask = np.zeros_like(bjd), np.zeros_like(bjd)
    for s in sect_ranges:
        # detrend this sector's LC
        g = np.in1d(sectors, s)

        # get units for exoplanet
        bjd_shift,fexo,efexo,ref_time = convert2exoplanet(bjd[g], fnorm[g], efnorm[g])

        # run initial optimization on the GP hyperparams
        model, map_soln, extras = build_model_0planets_SHO(bjd_shift, fexo, efexo, Prot)
        gp = extras['gp_pred']

        # clip outliers
        ##mask[g] = _sigma_clip(fexo, extras0)

        # redo optimization
        ##model,map_soln,extras = build_model_0planets(bjd_shift,fexo,Prot,
        ##                                             mask[g].astype(bool),
        ##                                             map_soln0)
        ##gp = np.zeros_like(fexo) + np.nan

        # detrend LC
        _,fdetrend[g],_ = convert2norm(bjd_shift, fexo-gp, efexo, ref_time)

    return fdetrend, map_soln, extras


def is_there_residual_rotation(ts, bjd, fdetrend, ef, sectors):
    '''
    Given a detrended light curve, check if there is still rotation. If there is,
    probably want to redo the detrending.
    '''
    # get most consecutive sectors
    sect_counts = [len(sr) for sr in ts.lc.sect_ranges]
    gs = np.in1d(sectors, ts.lc.sect_ranges[np.argmax(sect_counts)])

    # compute gls
    x, y = misc.bin_lc(bjd[gs], fdetrend[gs], 30)
    g = y != 0
    T = ts.lc.bjd_raw[gs].max() - ts.lc.bjd_raw[gs].min()
    gls = Gls((x[g], y[g], np.ones(g.sum())), fend=10, fbeg=1/T)

    # attribute the largest peak to rotation
    Prot = gls.best['P']

    # is there a strong signal in the detrended light curve?
    #outliers = np.invert(misc.sigma_clip(gls.power, offset=False))

    # check if a rotation period is likely detected (i.e. has large GLS power and 
    # sinusoidal model is favoured)
    theta = gls.best['amp'], gls.best['T0'], gls.best['P'], gls.best['offset']
    model = misc.sinemodel(bjd, *theta)
    _,_,dBIC = misc.DeltaBIC(fdetrend, ef, model, np.ones_like(model), k=4)
    rotation_detected = dBIC <= -10

    return rotation_detected, Prot

