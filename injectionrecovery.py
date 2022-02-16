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

global Tmagfname
Tmagfname = '%s/Tmag_file.csv'%cs.repo_dir


def compile_Tmags():
    # get all stars from the transit search
    fs = glob.glob('%s/MAST/TESS/TIC*/TESSLC_planetsearch'%(cs.repo_dir))
    ticids = np.sort([int(f.split('/TIC')[1].split('/')[0]) for f in fs])
    N = ticids.size

    # get tics and Tmags
    print('\nCompiling Tmag values for stars with a completed transit search...')
    ticsout, Tmags = np.zeros(N), np.zeros(N)
    for i,tic in enumerate(ticids):
        fname = '%s/MAST/TESS/TIC%i/TESSLC_planetsearch'%(cs.repo_dir,tic)
        ts = loadpickle(fname)
        if not ts.DONE:
            continue
        ticsout[i] = ts.tic
        Tmags[i] = ts.star.Tmag

    # save
    g = ticsout > 0
    df = pd.DataFrame(np.vstack([ticsout, Tmags]).T[g], columns=['TIC','Tmag'])
    df.to_csv(Tmagfname, index=False)    

    return ticids[g], Tmags[g]


def run_full_injection_recovery(Tmagmin, Tmagmax, use_20sec=False, overwrite=False, N1=500, N2=500):
    # group stars by Tmag (i.e. do inj-rec over Tmag bins)
    injrec = _get_injrec_object(Tmagmin, Tmagmax)
    df = pd.read_csv(Tmagfname)
    g = (df['Tmag'] >= injrec.Tmagmin) & (df['Tmag'] <= injrec.Tmagmax)
    injrec.tics, injrec.Tmags = np.ascontiguousarray(df['TIC'][g]), np.ascontiguousarray(df['Tmag'][g])
    assert injrec.tics.size > 0

    # do injection-recovery
    kwargs = {'N1': int(N1), 'N2': int(N2), 'pltt': True, 'overwrite': overwrite}
    do_injection_recovery(injrec, **kwargs)
    
    # save results
    assert injrec.DONE
    injrec.pickleobject()
   


def _get_injrec_object(Tmagmin, Tmagmax):
    '''
    Open an existing injrec object for this Tmag range if it exists. 
    Otherwise, create a new one and start from scratch.
    '''
    fname = ('injrec_Tmag_%.2f_%.2f'%(Tmagmin, Tmagmax)).replace('.','d')
    fname_full = '%s/MAST/TESS/%s'%(cs.repo_dir, fname)
    if os.path.exists(fname_full):
        injrec = loadpickle(fname_full)
    else:
        injrec = injection_recovery(Tmagmin, Tmagmax, fname)
        injrec.DONE1, injrec.DONE = False, False
    return injrec 


def bin_stars_Tmag(Tmagmin, Tmagmax, ticids):
    '''
    Read in Tmags and only keep stars within the desired range of TESS 
    magnitudes.
    '''
    assert Tmagmin < Tmagmax
    
    ticsout, Tmags = np.zeros(0), np.zeros(0)
    for i,tic in enumerate(ticids):
        fname = '%s/MAST/TESS/TIC%i/TESSLC_planetsearch'%(cs.repo_dir,tic)
        assert os.path.exists(fname)
        ts = loadpickle(fname)
        if (ts.star.Tmag >= Tmagmin) & (ts.star.Tmag <= Tmagmax):
            ticsout = np.append(ticsout, tic)
            Tmags = np.append(Tmags, ts.star.Tmag)

    return ticsout, Tmags
        


def do_injection_recovery(injrec, N1=500, N2=500, overwrite=True, pltt=True):
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
        _run_injection_recovery_iter2(injrec, N2=N2, pltt=pltt)



def _run_injection_recovery_iter1(injrec, N1=500):
    # first, uniformly sample planets in the P-Rp parameter space
    N1 = int(N1)
    P, dT0, Rp, b = sample_planets_uniform(N1)

    # for every planet, run the TLS and flag planet as recovered or not
    injrec_results = np.zeros((N1,8))
    T0, snr, sde = np.zeros(N1), np.zeros(N1), np.zeros(N1)
    for i in range(N1):

        print('%.3f (first set)'%(i/N1))

        # get star
        tic = np.random.choice(injrec.tics)
        ts = loadpickle('%s/MAST/TESS/TIC%i/TESSLC_planetsearch'%(cs.repo_dir,tic))
        while not hasattr(ts.lc, 'efnorm_rescaled'):
            tic = np.random.choice(injrec.tics)
            ts = loadpickle('%s/MAST/TESS/TIC%i/TESSLC_planetsearch'%(cs.repo_dir,tic))
        clean_injrec_lc(injrec, ts)
        T0[i] = ts.lc.bjd[0] + dT0[i]
        snr[i] = misc.estimate_snr(ts, P[i], T0[i], Rp[i])
        injrec_results[i,:6] = tic, P[i], T0[i], Rp[i], b[i], snr[i]

        # inject planet
        inject_custom_planet(injrec, ts, P[i], Rp[i], b=b[i], T0=T0[i])
        
        # run TLS
        print('\nRunning injection-recovery on TIC %i (P=%.3f days, T0=%.3f, Rp=%.2f REarth, b=%.2f, S/N=%.1f)'%tuple(injrec_results[i,:6]))
        det, sde[i] = run_tls_Nplanets(injrec, ts)
        injrec_results[i,6:] = sde[i], det

    # compute sensitivity vs snr
    snr_grid_big = np.arange(0,30)
    snr_grid = snr_grid_big[1:] - np.diff(snr_grid_big)[0]/2
    sens_gridv1 = np.zeros(snr_grid.size)
    for i in range(snr_grid.size):
        g = (snr > snr_grid_big[i]) & (snr <= snr_grid_big[i+1])
        sens_gridv1[i] = injrec_results[g,7].sum()/g.sum() if g.sum() > 0 else np.nan
    
    # compute sensitivity vs sde
    sde_grid_big = np.arange(0,50)
    sde_grid = sde_grid_big[1:] - np.diff(sde_grid_big)[0]/2
    sens_gridv2 = np.zeros(sde_grid.size)
    for i in range(sde_grid.size):
        g = (sde > sde_grid_big[i]) & (sde <= sde_grid_big[i+1])
        sens_gridv2[i] = injrec_results[g,7].sum()/g.sum() if g.sum() > 0 else np.nan

    # fit the sensitivity curve with a Gamma CDF
    g = np.isfinite(sens_gridv2)
    popt,_ = curve_fit(_gammaCDF, sde_grid[g], sens_gridv2[g], p0=[15,.5])
    sde_model = np.linspace(0,np.nanmax(sde),1000)
    sens_modelv2 = _gammaCDF(sde_model, *popt)   
   
    # recompute sensitivity vs snr
    for i in range(snr_grid.size):
        g = (injrec_results[:,5] > snr_grid_big[i]) & (injrec_results[:,5] <= snr_grid_big[i+1])
        sens_gridv1[i] = injrec_results[g,7].sum()/g.sum() if g.sum() > 0 else np.nan

    # recompute sensitivity vs sde
    for i in range(sde_grid.size):
        g = (injrec_results[:,6] > sde_grid_big[i]) & (injrec_results[:,6] <= sde_grid_big[i+1])
        sens_gridv2[i] = injrec_results[g,7].sum()/g.sum() if g.sum() > 0 else np.nan

    # fit the sensitivity curve with a Gamma CDF
    injrec.snr_grid, injrec.sens_snr_grid = snr_grid, sens_gridv1
    injrec.snr_grid_big = snr_grid_big
    injrec.sde_grid, injrec.sens_sde_grid = sde_grid, sens_gridv2
    injrec.sde_grid_big = sde_grid_big

    g = np.isfinite(sens_gridv1)
    injrec.popt_snr,_ = curve_fit(_gammaCDF, snr_grid[g], sens_gridv1[g], p0=[15,.5])
    injrec.snr_model = np.linspace(0,np.nanmax(snr),1000)
    injrec.sens_snr_model = _gammaCDF(injrec.snr_model, *injrec.popt_snr)

    g = np.isfinite(sens_gridv2)
    injrec.popt_sde,_ = curve_fit(_gammaCDF, sde_grid[g], sens_gridv2[g], p0=[15,.5])
    injrec.sde_model = np.linspace(0,np.nanmax(sde),1000)
    injrec.sens_sde_model = _gammaCDF(injrec.sde_model, *injrec.popt_sde)

    # save results
    injrec.injrec_results = injrec_results
    injrec.DONE1 = True



def _run_injection_recovery_iter2(injrec, N2=500, pltt=True):
    assert injrec.DONE1
    assert not injrec.DONE

    # for every planet, run the TLS and flag planet as recovered or not
    N2 = int(N2)
    injrec_resultsv2 = np.zeros((N2,8))
    P, Rp, b = np.zeros(N2), np.zeros(N2), np.zeros(N2)
    T0, snr, sde = np.zeros(N2), np.zeros(N2), np.zeros(N2)
    for i in range(N2):

        print('%.3f (second set)'%(i/N2))

        # get star
        tic = np.random.choice(injrec.tics)
        ts = loadpickle('%s/MAST/TESS/TIC%i/TESSLC_planetsearch'%(cs.repo_dir,tic))
        while not hasattr(ts.lc, 'efnorm_rescaled'):
            tic = np.random.choice(injrec.tics)
            ts = loadpickle('%s/MAST/TESS/TIC%i/TESSLC_planetsearch'%(cs.repo_dir,tic))
        clean_injrec_lc(injrec, ts)

        # resample planets where the S/N is not very close to zero or one
        P[i], dT0, Rp[i], b[i] = sample_planets_weighted(ts, injrec.popt_snr, N=1)
        T0[i] = ts.lc.bjd[0] + dT0
        snr[i] = misc.estimate_snr(ts, P[i], T0[i], Rp[i])
        injrec_resultsv2[i,:6] = tic, P[i], T0[i], Rp[i], b[i], snr[i]

        # inject planet
        inject_custom_planet(injrec, ts, P[i], Rp[i], b=b[i], T0=T0[i])
        
        # run TLS
        print('\nRunning injection-recovery on TIC %i (P=%.3f days, T0=%.3f, Rp=%.2f REarth, b=%.2f, S/N=%.1f)'%tuple(injrec_resultsv2[i,:6]))
        det, sde[i] = run_tls_Nplanets(injrec, ts)
        injrec_resultsv2[i,6:] = sde[i], det

    # combine results
    injrec_results = np.vstack([injrec.injrec_results, injrec_resultsv2])
    
    # recompute sensitivity vs snr
    snr_grid, snr_grid_big = injrec.snr_grid, injrec.snr_grid_big
    sens_gridv1 = np.zeros_like(snr_grid)
    for i in range(snr_grid.size):
        g = (injrec_results[:,5] > snr_grid_big[i]) & (injrec_results[:,5] <= snr_grid_big[i+1])
        sens_gridv1[i] = injrec_results[g,7].sum()/g.sum() if g.sum() > 0 else np.nan
    
    # recompute sensitivity vs sde
    sde_grid, sde_grid_big = injrec.sde_grid, injrec.sde_grid_big
    sens_gridv2 = np.zeros_like(sde_grid)
    for i in range(sde_grid.size):
        g = (injrec_results[:,6] > sde_grid_big[i]) & (injrec_results[:,6] <= sde_grid_big[i+1])
        sens_gridv2[i] = injrec_results[g,7].sum()/g.sum() if g.sum() > 0 else np.nan

    # fit the sensitivity curve with a Gamma CDF
    injrec.snr_grid, injrec.sens_snr_grid = snr_grid, sens_gridv1
    injrec.snr_grid_big = snr_grid_big
    injrec.sde_grid, injrec.sens_sde_grid = sde_grid, sens_gridv2
    injrec.sde_grid_big = sde_grid_big

    g = np.isfinite(sens_gridv1)
    injrec.popt_snr,_ = curve_fit(_gammaCDF, snr_grid[g], sens_gridv1[g], p0=[15,.5])
    injrec.snr_model = np.linspace(0,np.nanmax(snr),1000)
    injrec.sens_snr_model = _gammaCDF(injrec.snr_model, *injrec.popt_snr)

    g = np.isfinite(sens_gridv2)
    injrec.popt_sde,_ = curve_fit(_gammaCDF, sde_grid[g], sens_gridv2[g], p0=[15,.5])
    injrec.sde_model = np.linspace(0,np.nanmax(sde),1000)
    injrec.sens_sde_model = _gammaCDF(injrec.sde_model, *injrec.popt_sde)

    # save results
    injrec.tics_inj, injrec.Ps, injrec.T0s, injrec.Rps, injrec.bs, injrec.snrs, injrec.sdes, injrec.recovered = injrec_results.T
    delattr(injrec, 'injrec_results')
    injrec.DONE = True
 
    # save sens vs S/N plot
    if pltt:
        plt.figure(figsize=(8,4))
        plt.step(injrec.sde_grid, injrec.sens_sde_grid, 'k-', lw=3)
        plt.plot(injrec.sde_model, injrec.sens_sde_model, '-', lw=2)
        plt.title('TIC %i'%ts.tic, fontsize=12)
        plt.ylabel('CDF', fontsize=12)
        plt.xlabel('S/N', fontsize=12)
        plt.savefig(('%s/MAST/TESS/senscurve_Tmag_%.2f_%.2f'%(cs.repo_dir, injrec.Tmagmin, injrec.Tmagmax)).replace('.','d')+'.png')
        plt.close('all')



def sample_planets_uniform(N=1e3):
    N = int(N)
    P = 10**np.random.uniform(np.log10(cs.Pgrid[0]), np.log10(cs.Pgrid[1]), N)
    dT0 = np.random.uniform(0, P, N)
    Rp = np.random.uniform(cs.Rpgrid[0], cs.Rpgrid[1], N)
    b = np.random.uniform(cs.bgrid[0], cs.bgrid[1], N)
    return P, dT0, Rp, b
        


def sample_planets_weighted(ts, popt_snr, N=1e3, border=0.02):
    assert 0 <= border <= 1

    N = int(N)
    out = np.zeros((0,4))
    while out.shape[0] < N:
        P = 10**np.random.uniform(np.log10(cs.Pgrid[0]), np.log10(cs.Pgrid[1]), N)
        dT0 = np.random.uniform(0, P, N)
        Rp = np.random.uniform(cs.Rpgrid[0], cs.Rpgrid[1], N)
        b = np.random.uniform(cs.bgrid[0], cs.bgrid[1], N)
        
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

    # remove planet candidates    
    #for i in np.where(ts.vetting.vetting_mask)[0]:


    

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
    injrec.argsinjected = np.append(np.copy(args[3:-1]), snr)

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




def run_tls_Nplanets(injrec, ts, Nmax=3, rtol=0.02):
    '''
    Run the Transit-Least-Squares search for multiple signals on an input 
    (detrended) light curve.
    '''
    is_detected = False
    for i,sect in enumerate(ts.lc.sect_ranges):
        
        g = np.in1d(ts.lc.sectors, sect)
        lc_input_raw = ts.lc.bjd[g], ts.lc.fdetrend[g], ts.lc.efnorm_rescaled[g]
        lc_input = ts.lc.bjd[g], injrec.finjected[g], ts.lc.efnorm_rescaled[g]
        slabel = '%i'%sect[0] if len(sect) == 1 else '%i-%i'%(min(sect),max(sect))

        # get maximum period for 2 transits on average
        Pmax,_,_ = dtg.get_Ntransit_vs_period(ts.tic, ts.lc.bjd[g], ts.lc.sectors[g], pltt=False)

        # run the tls and search for the injected signal
        print('\nRunning TLS for injection-recovery (sector(s) %s)\n'%(slabel))
            
        # run tls on this sector for light curves with and without the injected planet (i.e. null)
        results_raw = _run_tls(*lc_input_raw, ts.star.ab, period_max=float(Pmax))
        results = _run_tls(*lc_input, ts.star.ab, period_max=float(Pmax))

        # get highest peaks in the TLS of the null light curve
        g = results_raw['power'] >= 0  # TEMP??
        s = np.argsort(results_raw['power'][g])[::-1]
        Psrec_raw = results_raw['periods'][g][s][:int(Nmax)]

        # get highest peaks in the TLS
        ##g = results['power'] >= cs.SDEthreshold
        g = results['power'] >= 0  # TEMP??
        s = np.argsort(results['power'][g])[::-1]
        Psrec = results['periods'][g][s][:int(Nmax)]

        # check if the planet is recovered
        sdeP = np.max(results['power'][np.isclose(results['periods'], injrec.argsinjected[0], rtol=.05)])
        thetainj = injrec.argsinjected[0], injrec.argsinjected[-1], sdeP   # P,snr,sde
        is_detected += is_planet_detected(ts, thetainj, Psrec, Psrec_raw, rtol=rtol)

        detlabel = 'is' if is_detected else 'is not'
        print('\nInjected planet %s detected around TIC %i (P=%.3f days, S/N=%.1f)'%(detlabel, ts.tic, thetainj[0], thetainj[1]))

        # get SDE of the injected period (if P > Pmax then there's no SDE value)
        g = np.isclose(results['periods'], injrec.argsinjected[0], rtol=rtol)
        sde = np.nanmax(results['power'][g]) if g.sum() > 0 else np.nan

        # stop searching if the planet is found
        if is_detected:
            return is_detected, sde

    return is_detected, sde



def _run_tls(bjd, fdetrend, efnorm, ab, period_max=0):
    model = tls.transitleastsquares(bjd, fdetrend, efnorm)
    if period_max > 0:
        results = model.power(u=ab, period_max=float(period_max),
                              period_min=np.min(cs.Pgrid))
    else:
        results = model.power(u=ab, period_min=np.min(cs.Pgrid))
    return results



def is_planet_detected(ts, thetainj, Psrec, Psrec_raw, rtol=.02):
    '''
    Given the period of the injected planet, check if any of the top peaks 
    in the TLS are close to the injected period.
    '''    
    Pinj, snrinj, sdeinj = thetainj

    # get possible peaks to check
    Ppeaks = []
    for j in range(1,5):
        Ppeaks.append(Pinj*j)
        Ppeaks.append(Pinj/j)
    Ppeaks = np.sort(Ppeaks)

    # get rotation periods and harmonics to reject
    Prots = []
    for j in range(1,4):
        Prots.append(ts.star.Prot/j)
    Prots = np.sort(Prots)

    # check if there is a peak in the TLS that is not close to rotation
    is_detected = False
    for p in Psrec:
        is_detected += np.any((np.isclose(Ppeaks, p, rtol=rtol))) & np.all(np.invert(np.isclose(Prots, p, rtol=rtol))) & np.all(np.invert(np.isclose(Psrec_raw, p, rtol=rtol))) & (snrinj >= 1) & (sdeinj >= cs.SDEthreshold) & (vet_Prot_injrec(ts, p))

    return is_detected


def vet_Prot_injrec(ts, p, rtol=0.02):
    '''
    Check that the planet candidate is not close to Prot or a harmonic.
    '''
    # get rotation periods and harmonics to reject
    Prots = []
    for j in range(1,4):
        Prots.append(ts.star.Prot/j)
    Prots = np.sort(Prots)

    # check each POI
    return np.all(np.invert(np.isclose(Prots, p, rtol=rtol)))
