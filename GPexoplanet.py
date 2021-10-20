from imports import *
import exoplanet as xo
import pymc3 as pm
import aesara_theano_fallback.tensor as tt
import pymc3_ext as pmx
from celerite2.theano import terms, GaussianProcess
import pickle, pdb
import astropy.units as u
import get_tess_data as gtd


def build_model_0planets(x, y, mask=None, start=None):

    if mask is None:
        mask = np.ones(len(x), dtype=bool)

    with pm.Model() as model:

        # Parameters for the stellar properties
        mean = pm.Normal("mean", mu=0.0, sd=10.0)

        # Transit jitter & GP parameters
        log_sigma_lc = pm.Normal(
            "log_sigma_lc", mu=np.log(np.std(y[mask])), sd=10
        )
        log_rho_gp = pm.Normal("log_rho_gp", mu=0, sd=10)
        log_sigma_gp = pm.Normal(
            "log_sigma_gp", mu=np.log(np.std(y[mask])), sd=10
        )

        resid = y[mask]

        # GP model for the light curve
        kernel = terms.SHOTerm(
            sigma=tt.exp(log_sigma_gp),
            rho=tt.exp(log_rho_gp),
            Q=1 / np.sqrt(2),
        )
        gp = GaussianProcess(kernel, t=x[mask], yerr=tt.exp(log_sigma_lc))
        gp.marginal("gp", observed=resid)

        # Fit for the maximum a posteriori parameters, I've found that I can get
        # a better solution by trying different combinations of parameters in turn
        if start is None:
            start = model.test_point
        map_soln = pmx.optimize(start=start,
                                vars=[log_sigma_lc, log_sigma_gp, log_rho_gp])
        map_soln = pmx.optimize(start=map_soln, vars=[mean])
        map_soln = pmx.optimize(start=map_soln,
                                vars=[log_sigma_lc, log_sigma_gp, log_rho_gp])
        map_soln = pmx.optimize(start=map_soln)

        extras = dict(
            zip(
                ["gp_pred"],
                pmx.eval_in_model([gp.predict(resid)], map_soln),
            )
        )

    return model, map_soln, extras




def build_model_1planet(x, y, texp, theta, mask=None, start=None):

    M_star, R_star, T0b, Pb, Zb = theta
    assert len(M_star) == 2
    assert len(R_star) == 2
    assert len(T0b) == 2

    if mask is None:
        mask = np.ones(len(x), dtype=bool)

    with pm.Model() as model:

        # Parameters for the stellar properties
        mean = pm.Normal("mean", mu=0.0, sd=10.0)
        u_star = xo.QuadLimbDark("u_star")
        star = xo.LimbDarkLightCurve(u_star)

        # Stellar parameters
        BoundedNormal = pm.Bound(pm.Normal, lower=0, upper=3)
        m_star = BoundedNormal("m_star", mu=M_star[0], sd=M_star[1])
        r_star = BoundedNormal("r_star", mu=R_star[0], sd=R_star[1])

        # Orbital parameters for the planets
        t0 = pm.Normal("t0", mu=T0b[0], sd=T0b[1])
        log_period = pm.Normal("log_period", mu=np.log(Pb), sd=1)
        period = pm.Deterministic("period", tt.exp(log_period))

        # Fit in terms of transit depth (assuming b<1)
        b = pm.Uniform("b", lower=0, upper=1)
        log_depth = pm.Normal("log_depth", mu=np.log(Zb), sigma=2.0)
        ror = pm.Deterministic("ror",
                               star.get_ror_from_approx_transit_depth(
                                   1e-3 * tt.exp(log_depth), b
                               ),
        )
        r_pl = pm.Deterministic("r_pl", ror * r_star)

        ecs = pmx.UnitDisk("ecs", testval=np.array([0.01, 0.0]))
        ecc = pm.Deterministic("ecc", tt.sum(ecs ** 2))
        omega = pm.Deterministic("omega", tt.arctan2(ecs[1], ecs[0]))
        xo.eccentricity.kipping13("ecc_prior", fixed=True, observed=ecc)

        # Transit jitter & GP parameters
        log_sigma_lc = pm.Normal(
            "log_sigma_lc", mu=np.log(np.std(y[mask])), sd=10
        )
        log_rho_gp = pm.Normal("log_rho_gp", mu=0, sd=10)
        log_sigma_gp = pm.Normal(
            "log_sigma_gp", mu=np.log(np.std(y[mask])), sd=10
        )

        # Orbit model
        orbit = xo.orbits.KeplerianOrbit(
            r_star=r_star,
            m_star=m_star,
            period=period,
            t0=t0,
            b=b,
            ecc=ecc,
            omega=omega,
        )

        # Compute the model light curve
        light_curves = (
            star.get_light_curve(orbit=orbit, r=r_pl, t=x[mask], texp=texp)
            * 1e3
        )
        light_curve = tt.sum(light_curves, axis=-1) + mean
        resid = y[mask] - light_curve

        # GP model for the light curve
        kernel = terms.SHOTerm(
            sigma=tt.exp(log_sigma_gp),
            rho=tt.exp(log_rho_gp),
            Q=1 / np.sqrt(2),
        )
        gp = GaussianProcess(kernel, t=x[mask], yerr=tt.exp(log_sigma_lc))
        gp.marginal("gp", observed=resid)
        #         pm.Deterministic("gp_pred", gp.predict(resid))

        # Compute and save the phased light curve models
        phase_lc = np.linspace(-.3, .3, 100)
        pm.Deterministic(
            "lc_pred",
            1e3
            * star.get_light_curve(
                orbit=orbit, r=r_pl, t=t0 + phase_lc, texp=texp
            )[..., 0],
        )

        # Fit for the maximum a posteriori parameters, I've found that I can get
        # a better solution by trying different combinations of parameters in turn
        if start is None:
            start = model.test_point
        map_soln = pmx.optimize(
            start=start, vars=[log_sigma_lc, log_sigma_gp, log_rho_gp]
        )
        map_soln = pmx.optimize(start=map_soln, vars=[log_depth])
        map_soln = pmx.optimize(start=map_soln, vars=[b])
        map_soln = pmx.optimize(start=map_soln, vars=[log_period, t0])
        map_soln = pmx.optimize(start=map_soln, vars=[u_star])
        map_soln = pmx.optimize(start=map_soln, vars=[log_depth])
        map_soln = pmx.optimize(start=map_soln, vars=[b])
        map_soln = pmx.optimize(start=map_soln, vars=[ecs])
        map_soln = pmx.optimize(start=map_soln, vars=[mean])
        map_soln = pmx.optimize(
            start=map_soln, vars=[log_sigma_lc, log_sigma_gp, log_rho_gp]
        )
        map_soln = pmx.optimize(start=map_soln)

        extras = dict(
            zip(
                ["light_curves", "gp_pred"],
                pmx.eval_in_model([light_curves, gp.predict(resid)], map_soln),
            )
        )

    return model, map_soln, extras

    


def _sigma_clip(fT, extras, sig=5):
    sig = 5
    mod = extras["gp_pred"] + extras["mean"]
    resid = fT - mod
    rms = np.sqrt(np.median(resid ** 2))
    mask = np.abs(resid) < sig*rms
    return mask



def convert2exoplanet(bjd, fnorm, efnorm):
    ref_time = np.nanmean([np.nanmax(bjd), np.nanmin(bjd)])
    return bjd-ref_time, 1e3*(fnorm-1), 1e3*efnorm, ref_time



def convert2norm(bjd_shift, fexo, efexo, ref_time):
    return bjd_shift+ref_time, 1e-3*fexo+1, 1e-3*efexo




def detrend_light_curve(bjd, fnorm, efnorm):
    # get units for exoplanet
    bjd_shift,fexo,efexo,ref_time = convert2exoplanet(bjd, fnorm, efnorm)

    # run initial optimization on the GP hyperparams
    model0, map_soln0, extras0 = build_model_0planets(bjd_shift, fexo)

    # clip outliers
    mask = _sigma_clip(fexo, extras0)

    # redo optimization
    model,map_soln,extras = build_model_0planets(bjd_shift,fexo,mask,map_soln0)

    # detrend LC
    bjd,fdetrend,efnorm = convert2norm(bjd_shift, fexo-extras['gp_pred'],
                                       efexo, ref_time)
    return bjd, fdetrend, efnorm, model, map_soln, extras




if __name__ == '__main__':
    bjd, flux, eflux, sectors, qual_flags, texps = \
        gtd.read_TESS_data(318022259, minsector=20, maxsector=20)
    bjd_shift, fexo,_, ref_time = gtd.convert2exoplanet(bjd, flux, eflux)

    #theta = [1.094, 0.039], [1.10, 0.023], [-13.7439,1e-3], 6.2706, .302
    theta = [.53, .053], [.53, .053], [2458844.381227-ref_time,1e-3], 6.2222, 1.847
    model, map_soln, extras = build_model_1planet(bjd_shift,fexo,texps.mean(),theta)

