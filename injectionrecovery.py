import numpy as np
import pylab as plt
import batman, misc, os, pdb
from scipy.state import gamma
from scipy.optimize import curve_fit
from tls_object import *
import constants as cs
import transitleastsquares as tls
import define_tls_grid as dtg


def run_full_injection_recovery(tic, use_20sec=False):
    fname = '%s/MAST/TESS/TIC%i/TESSLC_planetsearch'%(cs.repo_dir,tic)
    assert os.path.exists(fname)
    ts = loadpickle(fname)
    
    # remove planet candidate signals from the TLS search
    clean_injrec_lc(ts)

    # do injection-recovery
    run_injection_recovery(ts)
    
    # save results
    ts.pickleobject()
    


def clean_injrec_lc(ts):
    '''
    Clean the input light curve by removing planet candidates from the TLS 
    search and defining injection-recovery variables. 
    '''
    ts.injrec.bjd = np.copy(ts.lc.bjd)
    ts.injrec.fclean = np.copy(ts.lc.fdetrend)
    ts.injrec.efnorm = np.copy(ts.lc.efnorm)
    ts.injrec.sectors = np.copy(ts.lc.sectors)
    ts.injrec.sect_ranges = np.copy(ts.lc.sect_ranges)

    # remove planet candidates    
    #for i in np.where(ts.vetting.vetting_mask)[0]:
        


def run_injection_recovery(ts, N1=500, N2=500):
    '''
    Sample planets from a grid, inject them into the cleaned light curve, and 
    attempt to recover them using the TLS.
    '''
    # first, uniformly sample planets in the P-Rp parameter space
    N1 = int(N1)
    P, dT0, Rp, b = sample_planets_uniform(N1)
    T0 = ts.lc.bjd[0] + dT0
    snr = estimate_snr(ts, P, Rp)

    # for every planet, run the TLS and flag planet as recovered or not
    injrec_results = np.zeros((N1+N2,5))
    for i in range(N1):
        injrec_results[i,:4] = P[i], Rp[i], b[i], snr[i]
        # inject planet
        inject_custom_planet(ts, P[i], Rp[i], b=b[i], T0=T0[i])
        # run TLS
        injrec_results[i,4] = int(run_tls_Nplanets(ts))

    # compute sensitivity vs snr
    snr_grid = np.linspace(0,30,20)
    sens_grid = np.zeros(snr_grid.size-1)
    for i in range(snr_grid.size-1):
        g = (snr >= snr_grid[i]) & (snr <= snr_grid[i+1])
        sens_grid[i] = injrec_results[g,4].sum()/g.sum() if g.sum() > 0 else np.nan
    
    # fit the sensitivity curve with a Gamma CDF
    popt,_ = curve_fit(_gammaCDF, snr_grid, sens_grid)
    snr_model = np.linspace(0,snr.max(),1000)
    sens_model = _gammaCDF(snr_model, *popt)

    
    plt.plot(snrgrid, sensgrid, 'o')
    plt.plot(snr_model, sens_model, '-')
    plt.show()

    
    # resample planets where the S/N is not very close to zero or one
    N2 = int(N2)
    P, dT0, Rp, b = sample_planets_weighted(ts, popt, N=N2)
    T0 = ts.lc.bjd[0] + dT0
    snr = estimate_snr(ts, P, Rp)

    # for every planet, run the TLS and flag planet as recovered or not
    injrec_results = np.zeros((N2,5))
    for i in range(N2):
        injrec_results[N1+i,:4] = P[i], Rp[i], b[i], snr[i]
        # inject planet
        inject_custom_planet(ts, P[i], Rp[i], b=b[i], T0=T0[i])
        # run TLS
        injrec_results[N1+i,4] = int(run_tls_Nplanets(ts))

    # save results
    ts.injrec.Ps, ts.injrec.Rps, ts.injrec.bs, ts.injrec.snrs, ts.injrec.recovered = injrec_results.T

    


def sample_planet_uniform(N=1e3):
    N = int(N)
    P = 10**np.random.uniform(np.log10(cs.Pgrid[0]), np.log10(cs.Pgrid[1]), N)
    dT0 = np.random.uniform(0, P, N)
    Rp = np.random.uniform(cs.Rpgrid[0], cs.Rpgrid[1], N)
    b = np.random.uniform(cs.bgrid[0], cs.bgrid[1], N)
    return P, dT0, Rp, b
        


def sample_planet_weighted(ts, popt, N=1e3, border=0.02):
    assert 0 <= border <= 1

    N = int(N)
    out = np.zeros((0,4))
    while out.shape[0] < N:
        P = 10**np.random.uniform(np.log10(cs.Pgrid[0]), np.log10(cs.Pgrid[1]), N)
        dT0 = np.random.uniform(0, P, N)
        Rp = np.random.uniform(cs.Rpgrid[0], cs.Rpgrid[1], N)
        b = np.random.uniform(cs.bgrid[0], cs.bgrid[1], N)
        
        # get each planet's snr and corresponding sensitivity
        snr = estimate_snr(ts, P, Rp)
        sens = _gammaCDF(snr, *popt)
        accept = (sens > border) & (sens < 1-border)
        out = np.vstack([out, np.array([P,dT0,Rp,b]).T[accept]])
        
    # return only the requested number of parameters
    P, dT0, Rp, b = out[:N].T
    return P, dT0, Rp, b



def estimate_snr(ts, P, Rp):
    Z = (misc.Rearth2m(Rp) / misc.Rsun2m(ts.star.Rs))**2
    sig = np.median(ts.injrec.efnorm)
    dT = 27 * np.max([len(s) for s in ts.lc.sect_ranges])
    return Z/sig * np.sqrt(dT/P)
    


def _gammaCDF(snr, a, s):
    return gamma.cdf(snr, a, loc=1, scale=s)

    

def inject_custom_planet(ts, P, RpRearth, b=0, ecc=0, T0=np.nan, seed=np.nan):
    '''
    Inject one planet with user-defined parameters into an existing light
    curve contained within a tls object.
    '''
    # get transit model
    if np.isfinite(seed):
        np.random.seed(int(seed))
    if np.isnan(T0):
        T0 = np.random.uniform(ts.lc.bjd.min(), ts.lc.bjd.max())
    omegadeg = float(np.random.uniform(0,360))
    args = ts.injrec.bjd, ts.star.Ms, ts.star.Rs, P, float(T0), RpRearth, b, ecc, omegadeg, ts.star.ab
    ts.injrec.injected_model = transit_model(*args)
    ts.injrec.argsinjected = np.copy(args[3:-1])

    # inject transiting planet
    ts.injrec.finjected = ts.injrec.fclean * ts.injrec.injected_model




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




def run_tls_Nplanets(ts, Nmax=3):
    '''
    Run the Transit-Least-Squares search for multiple signals on an input 
    (detrended) light curve.
    '''
    # get approximate stellar parameters
    p = tls.catalog_info(TIC_ID=ts.tic)
    ts.star.ab, ts.star.Ms, ts.star.Ms_min, ts.star.Ms_max, ts.star.Rs, ts.star.Rs_min, ts.star.Rs_max = p

    # plot Ntransits vs period
    _=dtg.get_Ntransit_vs_period(ts.tic, ts.injrec.bjd, ts.injrec.sectors)

    is_detected = False
    for i,s in enumerate(ts.injrec.sect_ranges):
        
        g = np.in1d(ts.injrec.sectors, s)
        lc_input = ts.injrec.bjd[g], ts.injrec.finjected[g], ts.injrec.efnorm[g]
        slabel = '%i'%s[0] if len(s) == 1 else '%i-%i'%(min(s),max(s))

        # get maximum period for 2 transits on average
        Pmax,_,_ = dtg.get_Ntransit_vs_period(ts.tic, ts.injrec.bjd[g], ts.injrec.sectors[g], pltt=False)

        # run the tls and search for the injected signal
        print('\nRunning TLS for injection-recovery (sector(s) %s)\n'%(slabel))
            
        # run tls on this sector
        results = _run_tls(*lc_input, ts.star.ab, period_max=float(Pmax))

        # get highest peaks in the TLS
        s = np.argsort(ts.tls.results_1_s4['power'])[::-1]
        Psrec = ts.tls.results_1_s4['periods'][s][:int(Nmax)]
        
        # check if the planet is recovered
        is_detected += is_planet_detected(ts.injrec.argsinjected[0], Psrec)

        # stop searching if the planet is found
        if is_detected:
            return is_detected

    return is_detected



def _run_tls(bjd, fdetrend, efnorm, ab, period_max=0):
    model = tls.transitleastsquares(bjd, fdetrend, efnorm)
    if period_max > 0:
        results = model.power(u=ab, period_max=period_max)
    else:
        results = model.power(u=ab)
    return results



def is_planet_detected(Pinj, Psrec, rtol=.05):
    '''
    Given the period of the injected planet, check if any of the top peaks 
    in the TLS are close to the injected period.
    '''    
    # get possible peaks to check
    Ppeaks = []
    for j in range(2,10):
        Ppeaks.append(Pinj*j)
        Ppeaks.append(Pinj/j)
    Ppeaks = np.sort(Ppeaks)

    # check if there is a peak in the TLS
    is_detected = False
    for p in Psrec:
        is_detected += np.any(np.isclose(Ppeaks, p, rtol=rtol))
        
    return is_detected
