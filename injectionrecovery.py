import numpy as np
import pylab as plt
import pandas as pd
import batman, misc, os, glob, pdb
from scipy.stats import gamma
from scipy.optimize import curve_fit
from tls_object import *
from injrec_object import *
import constants as cs
import transitleastsquares as tls
import define_tls_grid as dtg
import planet_vetting as pv
from planet_search import mask_transits

global Tmagfname
Tmagfname = '%s/Tmag_file.csv'%cs.repo_dir


def run_full_injection_recovery(Tmagmin, Tmagmax, use_20sec=False, overwrite=False, N1=500, N2=500):
    # group stars by Tmag (i.e. do inj-rec over Tmag bins on the HPC)
    injrec = get_injrec_object(Tmagmin, Tmagmax)
    df = pd.read_csv(Tmagfname)
    g = (df['Tmag'] >= injrec.Tmagmin) & (df['Tmag'] <= injrec.Tmagmax)
    injrec.tics_unique = np.ascontiguousarray(df['TIC'][g])
    assert injrec.tics_unique.size > 0

    # do injection-recovery on these stars
    kwargs = {'N1': int(N1), 'N2': int(N2), 'overwrite': overwrite}
    do_injection_recovery(injrec, **kwargs)

    # save results
    assert injrec.DONE
    injrec.pickleobject()
   


def get_injrec_object(Tmagmin, Tmagmax):
    '''
    Open an existing injrec object for this Tmag range if it exists. 
    Otherwise, create a new one and start from scratch.
    '''
    assert Tmagmin < Tmagmax
    fname = ('injrec_Tmag_%.3f_%.3f'%(Tmagmin, Tmagmax)).replace('.','d')
    fname_full = '%s/MAST/TESS/%s'%(cs.repo_dir, fname)
    if os.path.exists(fname_full):
        injrec = loadpickle(fname_full)
    else:
        injrec = injection_recovery(Tmagmin, Tmagmax, fname)
        injrec.DONE1, injrec.DONE = False, False
    return injrec
        


def do_injection_recovery(injrec, N1=500, N2=500, overwrite=True):
    '''
    Sample planets from a grid, inject them into a cleaned light curve, and 
    attempt to recover them using the TLS. Only use light curves for stars that
    are given in the list argument tics.
    '''
    # do it twice so we can save in between
    if not injrec.DONE1 or overwrite:
        _run_injection_recovery_iter1(injrec, N1=N1)
        injrec.pickleobject()
    if not injrec.DONE or overwrite:
        _run_injection_recovery_iter2(injrec, N2=N2)



def _run_injection_recovery_iter1(injrec, N1=500):
    # first, uniformly sample planets in the P-Rp parameter space
    N1 = int(N1)
    P, dT0, Rp, b = sample_planets_uniform(N1)

    # for every planet, run the TLS and flag planet as recovered or not
    # {tic,Ms,Rs,Teff,Tmag,fluxerr, P,F,T0,Rp,b,snr,sde, recovered}
    injrec_results = np.zeros((N1, 14))
    FP_results = np.zeros((0,9))
    T0, Fs, snr, sde = np.zeros(N1), np.zeros(N1), np.zeros(N1), np.zeros(N1)
    for i in range(N1):

        print('%.3f (first set)'%(i/N1))

        # get star
        tic = np.random.choice(injrec.tics_unique)
        ts = loadpickle('%s/MAST/TESS/TIC%i/TESSLC_planetsearch'%(cs.repo_dir,tic))
        count = 0
        while ts.DONEcheck_version != cs.DONEcheck_version:
            tic = np.random.choice(injrec.tics_unique)
            ts = loadpickle('%s/MAST/TESS/TIC%i/TESSLC_planetsearch'%(cs.repo_dir,tic))
            if count < 1000:
                count += 1
            else:
                raise ValueError('No TIC is available.')

        # save stellar paramaters
        flux_err = np.nanmedian(ts.lc.efnorm_rescaled)
        injrec_results[i,:6] = tic, ts.star.Ms, ts.star.Rs, ts.star.Teff, ts.star.Tmag, flux_err
            
        clean_injrec_lc(injrec, ts)

        # save planet parameters
        T0[i] = ts.lc.bjd[0] + dT0[i]
        Fs[i] = misc.compute_instellation(ts.star.Teff, ts.star.Rs, ts.star.Ms, P[i])
        snr[i] = misc.estimate_snr(ts, P[i], T0[i], Rp[i])
        injrec_results[i,6:12] = P[i], Fs[i], T0[i], Rp[i], b[i], snr[i]

        # inject planet
        inject_custom_planet(injrec, ts, P[i], Rp[i], b=b[i], T0=T0[i])
        
        # run TLS
        print('\nRunning injection-recovery on TIC %i (P=%.3f days, F=%.1f FEarth, T0=%.3f, Rp=%.2f REarth, b=%.2f, S/N=%.1f)'%tuple(np.append(tic,injrec_results[i,6:12])))
        det, sde[i], FPdict = run_tls_Nplanets_and_vet(injrec, ts)
        injrec_results[i,12:] = sde[i], det

        # save FPs
        NFP = FPdict['Ps'].size
        for n in range(NFP):
            theta = np.array([FPdict[k][n] for k in np.sort(list(FPdict.keys()))])
            FP_results = np.append(FP_results, theta.reshape(1,9), axis=0)
        
    # compute sensitivity vs snr to used for weighted sampling in iter2
    injrec.snr_grid = np.arange(0,30)
    injrec.snr_grid_v2 = injrec.snr_grid[1:] - np.diff(injrec.snr_grid)[0]/2
    injrec.sens_grid = np.zeros(injrec.snr_grid.size-1)
    for i in range(injrec.snr_grid.size-1):
        g = (snr > injrec.snr_grid[i]) & (snr <= injrec.snr_grid[i+1])
        injrec.sens_grid[i] = injrec_results[g,-1].sum()/g.sum() if g.sum() > 0 else np.nan
    
    g = np.isfinite(injrec.sens_grid)
    injrec.popt_snr,_ = curve_fit(_gammaCDF, injrec.snr_grid_v2[g], injrec.sens_grid[g], p0=[15,.5])

    # save results
    injrec.injrec_results = injrec_results
    injrec.fp.FP_results = FP_results
    injrec.DONE1 = True



def _run_injection_recovery_iter2(injrec, N2=500):
    assert injrec.DONE1

    N2 = int(N2)

    # for every planet, run the TLS and flag planet as recovered or not
    # {tic,Ms,Rs,Teff,Tmag,fluxerr, P,F,T0,Rp,b,snr,sde, recovered}
    injrec_resultsv2 = np.zeros((N2,14))
    FP_resultsv2 = np.zeros((0,9))
    P, Rp, b = np.zeros(N2), np.zeros(N2), np.zeros(N2)
    T0, Fs, snr, sde = np.zeros(N2), np.zeros(N2), np.zeros(N2), np.zeros(N2)
    for i in range(N2):

        print('%.3f (second set)'%(i/N2))

        # get star
        tic = np.random.choice(injrec.tics_unique)
        ts = loadpickle('%s/MAST/TESS/TIC%i/TESSLC_planetsearch'%(cs.repo_dir,tic))
        count = 0
        while ts.DONEcheck_version != cs.DONEcheck_version:
            tic = np.random.choice(injrec.tics_unique)
            ts = loadpickle('%s/MAST/TESS/TIC%i/TESSLC_planetsearch'%(cs.repo_dir,tic))
            if count < 1000:
                count += 1
            else:
                raise ValueError('No TIC is available.')

        # save stellar paramaters
        flux_err = np.nanmedian(ts.lc.efnorm_rescaled)
        injrec_resultsv2[i,:6] = tic, ts.star.Ms, ts.star.Rs, ts.star.Teff, ts.star.Tmag, flux_err
            
        clean_injrec_lc(injrec, ts)

        # resample planets where the S/N is not very close to zero or one
        P[i], dT0, Rp[i], b[i] = sample_planets_weighted(ts, injrec.popt_snr, N=1)
        T0[i] = ts.lc.bjd[0] + dT0
        snr[i] = misc.estimate_snr(ts, P[i], T0[i], Rp[i])
        Fs[i] = misc.compute_instellation(ts.star.Teff, ts.star.Rs, ts.star.Ms, P[i])
        injrec_resultsv2[i,6:12] = P[i], Fs[i], T0[i], Rp[i], b[i], snr[i]

        # inject planet
        inject_custom_planet(injrec, ts, P[i], Rp[i], b=b[i], T0=T0[i])
        
        # run TLS
        print('\nRunning injection-recovery on TIC %i (P=%.3f days, F=%.1f FEarth, T0=%.3f, Rp=%.2f REarth, b=%.2f, S/N=%.1f)'%tuple(np.append(tic,injrec_resultsv2[i,6:12])))
        det, sde[i], FPdict = run_tls_Nplanets_and_vet(injrec, ts)
        injrec_resultsv2[i,12:] = sde[i], det

        # save FPs
        NFP = FPdict['Ps'].size
        for n in range(NFP):
            theta = np.array([FPdict[k][n] for k in np.sort(list(FPdict.keys()))])
            FP_resultsv2 = np.append(FP_resultsv2, theta.reshape(1,9), axis=0)
        
    # combine results from iter1 and iter2
    injrec_results = np.vstack([injrec.injrec_results, injrec_resultsv2])
    FP_results = np.vstack([injrec.fp.FP_results, FP_resultsv2])

    # save results
    for i,s in enumerate(['tics','Mss','Rss','Teffs','Tmags','efluxes','Ps','Fs','T0s','Rps','bs','snrs','sdes','recovered']):
        setattr(injrec, s, injrec_results[:,i])
        
    # save FP results
    for i,k in enumerate(np.sort(list(FPdict.keys()))):
        setattr(injrec.fp, k, FP_results[:,i])
        
    # delete old stuff
    delattr(injrec.fp, 'FP_results')
    for s in ['injrec_results','argsinjected','bjd','fclean','finjected','efnorm','efnorm_rescaled','injected_model','sect_ranges','sectors','snr_grid','snr_grid_v2','sens_grid','tls','popt_snr']:
        try:
            delattr(injrec, s)
        except AttributeError:
            pass
            
    injrec.DONE = True



def sample_planets_uniform(N=1e3):
    N = int(N)
    P = 10**np.random.uniform(np.log10(cs.Pgrid[0]), np.log10(cs.Pgrid[1]), N)
    dT0 = np.random.uniform(0, P, N)
    Rp = np.random.uniform(cs.Rpgrid[0], cs.Rpgrid[1], N)
    b = abs(np.random.uniform(cs.bgrid[0], cs.bgrid[1], N))
    return P, dT0, Rp, b
        


def sample_planets_weighted(ts, popt_snr, N=1e3, border=0.02):
    assert 0 <= border <= 1

    N = int(N)
    out = np.zeros((0,4))
    while out.shape[0] < N:
        P = 10**np.random.uniform(np.log10(cs.Pgrid[0]), np.log10(cs.Pgrid[1]), N)
        dT0 = np.random.uniform(0, P, N)
        Rp = np.random.uniform(cs.Rpgrid[0], cs.Rpgrid[1], N)
        b = abs(np.random.uniform(cs.bgrid[0], cs.bgrid[1], N))
        
        # get each planet's snr and corresponding sensitivity
        snr = misc.estimate_snr(ts, P, ts.lc.bjd[0]+dT0, Rp)
        sens = _gammaCDF(snr, *popt_snr)
        accept = (sens > border) & (sens < 1-border)
        out = np.vstack([out, np.array([P,dT0,Rp,b]).T[accept]])
        
    # return only the requested number of parameters
    P, dT0, Rp, b = out[:N].T
    return P, dT0, Rp, b



def clean_injrec_lc(injrec, ts):
    '''
    Clean the input light curve by removing planet candidates from the TLS 
    search and defining injection-recovery variables. 
    '''
    injrec.bjd = np.copy(ts.lc.bjd)
    injrec.fclean = np.copy(ts.lc.fdetrend)
    injrec.efnorm = np.copy(ts.lc.efnorm)
    injrec.efnorm_rescaled = np.copy(ts.lc.efnorm_rescaled)
    injrec.sectors = np.copy(ts.lc.sectors)
    injrec.sect_ranges = np.copy(ts.lc.sect_ranges)

    

def _gammaCDF(snr, k, theta):
    return gamma.cdf(snr, k, loc=1, scale=theta)

    

def inject_custom_planet(injrec, ts, P, RpRearth, b=0, ecc=0, T0=np.nan, seed=np.nan):
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
    args = ts.lc.bjd, ts.star.Ms, ts.star.Rs, P, float(T0), RpRearth, b, ecc, omegadeg, ts.star.ab
    injrec.injected_model = transit_model(*args)
    snr = misc.estimate_snr(ts, P, float(T0), RpRearth)
    injrec.argsinjected = np.append(np.copy(args[3:-1]), snr) # {P,T0,rp,b,ecc,snr}

    # inject transiting planet
    injrec.finjected = injrec.fclean * injrec.injected_model



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



def run_tls_Nplanets_and_vet(injrec, ts, Nmax=3, rtol=0.02):
    '''
    Run the Transit-Least-Squares search for multiple signals on an input 
    (detrended) light curve. Return whether the in
    '''
    Pinj = injrec.argsinjected[0]
    is_detected = False
    for i,sect in enumerate(ts.lc.sect_ranges):

        # get data from this sector
        g = np.in1d(ts.lc.sectors, sect)
        lc_input = ts.lc.bjd[g], injrec.finjected[g], ts.lc.efnorm_rescaled[g]
        slabel = '%i'%sect[0] if len(sect) == 1 else '%i-%i'%(min(sect),max(sect))

        # get maximum period for Nmin transits on average
        Pmax,_,_ = dtg.get_Ntransit_vs_period(ts.tic, ts.lc.bjd[g], ts.lc.sectors[g], pltt=False)

        # run the tls and search for the injected signal
        iter_count = 1
        print('\nRunning TLS for injection-recovery %i (sector(s) %s)\n'%(iter_count,slabel))

        # run tls on this sector for light curves with and without the injected planet (i.e. null)
        results = _run_tls(*lc_input, ts.star.ab, period_max=float(Pmax))
        rp = misc.m2Rearth(misc.Rsun2m(results['rp_rs']*ts.star.Rs))
        results.snr = misc.estimate_snr(ts, results['period'], results['T0'], rp) if np.isfinite(results['period']) else np.nan
        setattr(ts.injrec.tls, 'results_%i_s%s'%(iter_count,slabel), results)

        # check if detected to save time
        if np.any(np.isclose([Pinj/2, Pinj, Pinj*2], results.period, rtol=rtol)):
            break

        # mask out found signal
        lc_input = mask_transits(*lc_input, getattr(ts.injrec.tls, 'results_%i_s%s'%(iter_count,slabel)))

        while (iter_count < cs.Nplanets_max) and (np.any(results.power >= cs.SDEthreshold)):

            # run tls on this sector
            iter_count += 1
            print('\nRunning TLS for planet %i (sector(s) %s)\n'%(iter_count,slabel))
            results = _run_tls(*lc_input, ts.star.ab, period_max=float(Pmax))
            setattr(ts.injrec.tls, 'results_%i_s%s'%(iter_count,slabel), results)

            # mask out found signal
            lc_input = mask_transits(*lc_input, getattr(ts.injrec.tls, 'results_%i_s%s'%(iter_count,slabel)))

    # vet OIs using the same criteria as in the transit search
    vet_planets(ts)

    # check if the injected planets is recovered (function also flags FPs)
    is_detected = is_planet_detected(ts, Pinj)

    detlabel = 'is' if is_detected else 'is not'
    print('\nInjected planet %s detected around TIC %i (P=%.3f days, S/N=%.1f)\n'%(detlabel, ts.tic, Pinj, injrec.argsinjected[-1]))
    if not is_detected:
        print(ts.injrec.vetting.POIs, ts.injrec.vetting.conditions_indiv, ts.injrec.vetting.conditions_defs, '\n')

    # get SDE of the injected period (if P > Pmax then there's no SDE value)
    g = np.isclose(results['periods'], Pinj, rtol=rtol)
    sde = np.nanmax(results['power'][g]) if g.sum() > 0 else np.nan

    # identify FP signals
    FPmask = ts.injrec.vetting.FP_mask * ts.injrec.vetting.vetting_mask
    FPdict = {'Ps': ts.injrec.vetting.POIs[FPmask],
              'T0s': ts.injrec.vetting.T0OIs[FPmask],
              'Ds': ts.injrec.vetting.DOIs[FPmask],
              'Zs': ts.injrec.vetting.ZOIs[FPmask],
              'rpRss': ts.injrec.vetting.rpRsOIs[FPmask],
              'SDEraws': ts.injrec.vetting.SDErawOIs[FPmask],
              'SDEs': ts.injrec.vetting.SDEOIs[FPmask],
              'snrs': ts.injrec.vetting.snrOIs[FPmask],
              'efluxes': np.repeat(np.nanmedian(ts.lc.efnorm_rescaled), FPmask.sum())}
        
    return is_detected, sde, FPdict
        

        
def _run_tls(bjd, fdetrend, efnorm, ab, period_max=0):
    model = tls.transitleastsquares(bjd, fdetrend, efnorm)
    if period_max > 0:
        results = model.power(u=ab, period_max=float(period_max),
                              period_min=np.min(cs.Pgrid))
    else:
        results = model.power(u=ab, period_min=np.min(cs.Pgrid))
    return results



def vet_planets(ts):
    '''
    Given the TLS results, vet planets using various metrics.
    '''
    pv.get_POIs(ts, injrec=True)
    pv.vet_SDE(ts, injrec=True)
    #pv.vet_snr(ts, injrec=True)   # snr<3: TOIs 2136
    #pv.vet_multiple_sectors(ts, injrec=True)
    pv.vet_odd_even_difference(ts, injrec=True)
    pv.vet_Prot(ts, injrec=True)
    pv.vet_tls_Prot(ts, injrec=True)
    pv.model_comparison(ts, injrec=True)
    pv.identify_conditions(ts, injrec=True)



def is_planet_detected(ts, Pinj, rtol=0.02):
    '''
    Given the list of OIs from the TLS, check which OIs are FPs and if any vetted planets are consistent 
    with the injected planets (i.e. is the injected planet detected?). 
    '''
    # also check multiples of the injected period 
    Ps_inj = []
    for j in range(1,5):
        Ps_inj.append(Pinj*j)
        Ps_inj.append(Pinj/j)
    Ps_inj = np.sort(np.unique(Ps_inj))

    # for each vetted OI, check which are consistent with Pinj and which are FPs
    ts.injrec.vetting.FP_mask = np.ones_like(ts.injrec.vetting.POIs).astype(bool)
    for i,passed in enumerate(ts.injrec.vetting.vetting_mask):
        if passed:
            ts.injrec.vetting.FP_mask[i] = np.invert(np.any(np.isclose(Ps_inj, ts.injrec.vetting.POIs[i], rtol=rtol)))

    # is the injected planet detected?
    is_detected = np.any(np.invert(ts.injrec.vetting.FP_mask))

    return is_detected



def compile_Tmags():
    # get all stars from the transit search
    fs = glob.glob('%s/MAST/TESS/TIC*/TESSLC_planetsearch'%(cs.repo_dir))
    ticids = np.sort([int(f.split('/TIC')[1].split('/')[0]) for f in fs])
    N = ticids.size

    # get tics and Tmags
    print('\nCompiling Tmag values for stars with a completed transit search...')
    ticsout, Tmags = np.zeros(N), np.zeros(N)
    for i,tic in enumerate(ticids):
        if i % 1e2 == 0: print(i/N)
        fname = '%s/MAST/TESS/TIC%i/TESSLC_planetsearch'%(cs.repo_dir,tic)
        try:
            ts = loadpickle(fname)
            if ts.DONE and ts.DONEcheck_version == cs.DONEcheck_version:
                ticsout[i] = ts.tic
                Tmags[i] = ts.star.Tmag
        except:
            pass

    # save
    g = ticsout > 0
    df = pd.DataFrame(np.vstack([ticsout, Tmags]).T[g], columns=['TIC','Tmag'])
    df.to_csv(Tmagfname, index=False)    

    return ticids[g], Tmags[g]
