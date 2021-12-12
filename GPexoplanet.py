import numpy as np
import pylab as plt
import exoplanet as xo
import pymc3 as pm
import aesara_theano_fallback.tensor as tt
import pymc3_ext as pmx
from celerite2.theano import terms, GaussianProcess
import pickle, pdb
import astropy.units as u


MAD = lambda x: np.median(abs(x-np.median(x)))


def build_model_0planets(x, y, Prot, mask=None, start=None):

    if mask is None:
        mask = np.ones(len(x), dtype=bool)

    with pm.Model() as model:

        # Parameters for the stellar properties
        mean = pm.Normal("mean", mu=0.0, sd=10.0)

        # Transit jitter & GP parameters
        #log_sigma_gp = pm.Normal("log_sigma_gp", mu=np.log(np.std(y[mask])), sd=1)
        #log_rho_gp = pm.Uniform("log_rho_gp", lower=10, upper=20)
        log_sigma_gp = pm.Uniform("log_sigma_gp", lower=-3, upper=np.log(np.std(y[mask])))
        log_rho_gp = pm.Normal("log_rho_gp", mu=np.log(Prot), sd=1)
        log_tau_gp = pm.Uniform("log_tau_gp", lower=np.log(30*Prot), upper=20)
        log_sigma_lc = pm.Normal("log_sigma_lc", mu=np.log(MAD(y[mask])), sd=1)

        resid = y[mask]

        # GP model for the light curve
        kernel = terms.SHOTerm(
            sigma=tt.exp(log_sigma_gp),
            rho=tt.exp(log_rho_gp),
            tau=tt.exp(log_tau_gp),
        )
        gp = GaussianProcess(kernel, t=x[mask], yerr=tt.exp(log_sigma_lc))
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
    sig = 5
    mod = extras["gp_pred"]
    resid = fT - mod
    rms = np.sqrt(np.median(resid ** 2))
    mask = np.abs(resid) < sig*rms
    return mask



def convert2exoplanet(bjd, fnorm, efnorm):
    ref_time = np.nanmean([np.nanmax(bjd), np.nanmin(bjd)])
    return bjd-ref_time, 1e3*(fnorm-1), 1e3*efnorm, ref_time



def convert2norm(bjd_shift, fexo, efexo, ref_time):
    return bjd_shift+ref_time, 1e-3*fexo+1, 1e-3*efexo




def detrend_light_curve(bjd, fnorm, efnorm, sectors, Prot):
    assert bjd.size == fnorm.size
    assert bjd.size == efnorm.size
    assert bjd.size == sectors.size

    fdetrend, mask = np.zeros_like(bjd), np.zeros_like(bjd)
    for s in np.unique(sectors):
        # detrend this sector's LC
        g = sectors == s

        # get units for exoplanet
        bjd_shift,fexo,efexo,ref_time = convert2exoplanet(bjd[g], fnorm[g], efnorm[g])

        # run initial optimization on the GP hyperparams
        model0, map_soln0, extras0 = build_model_0planets(bjd_shift, fexo, Prot)

        # clip outliers
        mask[g] = _sigma_clip(fexo, extras0)

        # redo optimization
        model,map_soln,extras = build_model_0planets(bjd_shift,fexo,Prot,
                                                     mask[g].astype(bool),
                                                     map_soln0)
        gp = np.zeros_like(fexo) + np.nan
        gp[mask[g].astype(bool)] = extras['gp_pred']

        # detrend LC
        _,fdetrend[g],_ = convert2norm(bjd_shift, fexo-gp, efexo, ref_time)

    return fdetrend, mask.astype(bool), map_soln, extras
