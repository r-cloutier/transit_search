import numpy as np
import pylab as plt
import pandas as pd
import batman, misc, os, glob, pdb, george
from scipy.stats import gamma
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from gls import Gls
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
    injrec = get_injrec_object(Tmagmin, Tmagmax)
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
   


def get_injrec_object(Tmagmin, Tmagmax):
    '''
    Open an existing injrec object for this Tmag range if it exists. 
    Otherwise, create a new one and start from scratch.
    '''
    fname = ('Protinjrec_Tmag_%.2f_%.2f'%(Tmagmin, Tmagmax)).replace('.','d')
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
    Sample rotation periods and photometric amplitudes from a grid, inject them
    into a light curve with known sampling and uncertainties, then attempt to
    recover them using the GLS.
    '''
    # do it twice so we can save in between
    if not injrec.DONE1 or overwrite:
        _run_injection_recovery_iter1(injrec, N1=N1)
        injrec.pickleobject()
    if not injrec.DONE or overwrite:
        _run_injection_recovery_iter2(injrec, N2=N2, pltt=pltt)



def _run_injection_recovery_iter1(injrec, N1=500):
    # first, uniformly sample a and Prot
    N1 = int(N1)
    amp, Prot, Gamma = sample_rotation_uniform(N1)

    # for every planet, run the TLS and flag planet as recovered or not
    injrec_results = np.zeros((N1,7))
    snr, gls_pwr = np.zeros(N1), np.zeros(N1)
    for i in range(N1):

        print('%.3f (first set)'%(i/N1))

        # get star
        tic = np.random.choice(injrec.tics)
        ts = loadpickle('%s/MAST/TESS/TIC%i/TESSLC_planetsearch'%(cs.repo_dir,tic))
        while not hasattr(ts.lc, 'efnorm_rescaled'):
            tic = np.random.choice(injrec.tics)
            ts = loadpickle('%s/MAST/TESS/TIC%i/TESSLC_planetsearch'%(cs.repo_dir,tic))
        clean_injrec_lc(injrec, ts)
        snr[i] = misc.estimate_Prot_snr(ts, amp[i], Prot[i])
        injrec_results[i,:5] = tic, amp[i], Prot[i], Gamma[i], snr[i]

        # inject rotation
        inject_rotation(injrec, ts, amp[i], Prot[i], Gamma=Gamma[i])
        
        # run GLS
        print('\nRunning injection-recovery on TIC %i (amp=%.1f ppt, Prot=%.1f days, Gamma=%.1f, S/N=%.1f)'%tuple(injrec_results[i,:5]))
        det, gls_pwr[i] = run_gls(injrec, ts, Prot[i])
        injrec_results[i,5:] = gls_pwr[i], det


    # compute sensitivity vs snr
    snr_grid_big = np.arange(0,30)
    snr_grid = snr_grid_big[1:] - np.diff(snr_grid_big)[0]/2
    sens_gridv1 = np.zeros(snr_grid.size)
    for i in range(snr_grid.size):
        g = (snr > snr_grid_big[i]) & (snr <= snr_grid_big[i+1])
        sens_gridv1[i] = injrec_results[g,6].sum()/g.sum() if g.sum() > 0 else np.nan
    
    # compute sensitivity vs gls power
    glspwr_grid_big = np.arange(51)/50
    glspwr_grid = glspwr_grid_big[1:] - np.diff(glspwr_grid_big)[0]/2
    sens_gridv2 = np.zeros(glspwr_grid.size)
    for i in range(glspwr_grid.size):
        g = (gls_pwr > glspwr_grid_big[i]) & (gls_pwr <= glspwr_grid_big[i+1])
        sens_gridv2[i] = injrec_results[g,6].sum()/g.sum() if g.sum() > 0 else np.nan

    # fit the sensitivity curve with a Gamma CDF
    g = np.isfinite(sens_gridv2)
    popt,_ = curve_fit(_gammaCDF, glspwr_grid[g], sens_gridv2[g], p0=[15,.5])
    glspwr_model = np.linspace(0,1,1000)
    sens_modelv2 = _gammaCDF(glspwr_model, *popt)   
   
    # recompute sensitivity vs snr
    for i in range(snr_grid.size):
        g = (injrec_results[:,4] > snr_grid_big[i]) & (injrec_results[:,4] <= snr_grid_big[i+1])
        sens_gridv1[i] = injrec_results[g,6].sum()/g.sum() if g.sum() > 0 else np.nan

    # recompute sensitivity vs gls power
    for i in range(glspwr_grid.size):
        g = (injrec_results[:,5] > glspwr_grid_big[i]) & (injrec_results[:,5] <= glspwr_grid_big[i+1])
        sens_gridv2[i] = injrec_results[g,6].sum()/g.sum() if g.sum() > 0 else np.nan

    # fit the sensitivity curve with a Gamma CDF
    injrec.snr_grid, injrec.sens_snr_grid = snr_grid, sens_gridv1
    injrec.snr_grid_big = snr_grid_big
    injrec.glspwr_grid, injrec.sens_glspwr_grid = glspwr_grid, sens_gridv2
    injrec.glspwr_grid_big = glspwr_grid_big

    g = np.isfinite(sens_gridv1)
    injrec.popt_snr,_ = curve_fit(_gammaCDF, snr_grid[g], sens_gridv1[g], p0=[15,.5])
    injrec.snr_model = np.linspace(0,np.nanmax(snr),1000)
    injrec.sens_snr_model = _gammaCDF(injrec.snr_model, *injrec.popt_snr)

    g = np.isfinite(sens_gridv2)
    injrec.popt_glspwr,_ = curve_fit(_gammaCDF, glspwr_grid[g], sens_gridv2[g], p0=[15,.5])
    injrec.glspwr_model = np.linspace(0,1,1000)
    injrec.sens_glspwr_model = _gammaCDF(injrec.glspwr_model, *injrec.popt_glspwr)

    # save results
    injrec.injrec_results = injrec_results
    injrec.DONE1 = True



def _run_injection_recovery_iter2(injrec, N2=500, pltt=True):
    assert injrec.DONE1
    assert not injrec.DONE

    # for every planet, run the TLS and flag planet as recovered or not
    N2 = int(N2)
    injrec_resultsv2 = np.zeros((N2,7))
    amp, Prot, Gamma = np.zeros(N2), np.zeros(N2), np.zeros(N2)
    snr, gls_pwr = np.zeros(N2), np.zeros(N2)
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
        amp[i], Prot[i], Gamma[i] = sample_rotation_weighted(ts, injrec.popt_snr, N=1)
        snr[i] = misc.estimate_Prot_snr(ts, amp[i], Prot[i])
        injrec_resultsv2[i,:5] = tic, amp[i], Prot[i], Gamma[i], snr[i]

        # inject rotation
        inject_rotation(injrec, ts, amp[i], Prot[i], Gamma=Gamma[i])
        
        # run GLS
        print('\nRunning injection-recovery on TIC %i (amp=%.1f ppt, Prot=%.1f days, Gamma=%.1f, S/N=%.1f)'%tuple(injrec_resultsv2[i,:5]))
        det, gls_pwr[i] = run_gls(injrec, ts, Prot[i])
        injrec_resultsv2[i,5:] = gls_pwr[i], det

    # combine results
    injrec_results = np.vstack([injrec.injrec_results, injrec_resultsv2])
    
    # recompute sensitivity vs snr
    snr_grid, snr_grid_big = injrec.snr_grid, injrec.snr_grid_big
    sens_gridv1 = np.zeros_like(snr_grid)
    for i in range(snr_grid.size):
        g = (injrec_results[:,4] > snr_grid_big[i]) & (injrec_results[:,4] <= snr_grid_big[i+1])
        sens_gridv1[i] = injrec_results[g,6].sum()/g.sum() if g.sum() > 0 else np.nan
    
    # recompute sensitivity vs gls power
    glspwr_grid, glspwr_grid_big = injrec.glspwr_grid, injrec.glspwr_grid_big
    sens_gridv2 = np.zeros_like(glspwr_grid)
    for i in range(glspwr_grid.size):
        g = (injrec_results[:,5] > glspwr_grid_big[i]) & (injrec_results[:,5] <= glspwr_grid_big[i+1])
        sens_gridv2[i] = injrec_results[g,6].sum()/g.sum() if g.sum() > 0 else np.nan

    # fit the sensitivity curve with a Gamma CDF
    injrec.snr_grid, injrec.sens_snr_grid = snr_grid, sens_gridv1
    injrec.snr_grid_big = snr_grid_big
    injrec.glspwr_grid, injrec.sens_glspwr_grid = glspwr_grid, sens_gridv2
    injrec.glspwr_grid_big = glspwr_grid_big

    g = np.isfinite(sens_gridv1)
    injrec.popt_snr,_ = curve_fit(_gammaCDF, snr_grid[g], sens_gridv1[g], p0=[15,.5])
    injrec.snr_model = np.linspace(0,np.nanmax(snr),1000)
    injrec.sens_snr_model = _gammaCDF(injrec.snr_model, *injrec.popt_snr)

    g = np.isfinite(sens_gridv2)
    injrec.popt_glspwr,_ = curve_fit(_gammaCDF, glspwr_grid[g], sens_gridv2[g], p0=[15,.5])
    injrec.glspwr_model = np.linspace(0,1,1000)
    injrec.sens_glspwr_model = _gammaCDF(injrec.glspwr_model, *injrec.popt_glspwr)

    # save results
    injrec.tics_inj, injrec.amps, injrec.Prots, injrec.Gammas, injrec.snrs, injrec.glspwrs, injrec.recovered = injrec_results.T
    delattr(injrec, 'injrec_results')
    injrec.DONE = True
 
    # save sens vs S/N plot
    if pltt:
        plt.figure(figsize=(8,4))
        plt.step(injrec.glspwr_grid, injrec.sens_glspwr_grid, 'k-', lw=3)
        plt.plot(injrec.glspwr_model, injrec.sens_glspwr_model, '-', lw=2)
        plt.title('TIC %i'%ts.tic, fontsize=12)
        plt.ylabel('CDF', fontsize=12)
        plt.xlabel('S/N', fontsize=12)
        plt.savefig(('%s/MAST/TESS/senscurve_Tmag_%.2f_%.2f'%(cs.repo_dir, injrec.Tmagmin, injrec.Tmagmax)).replace('.','d')+'.png')
        plt.close('all')



def sample_rotation_uniform(N=1e3):
    N = int(N)
    amp = 10**np.random.uniform(np.log10(cs.ampgrid[0]),
                                np.log10(cs.ampgrid[1]), N)
    Prot = 10**np.random.uniform(np.log10(cs.Protgrid[0]),
                                 np.log10(cs.Protgrid[1]), N)
    Gamma = 10**np.random.uniform(0, 1, N)
    return amp, Prot, Gamma
        


def sample_rotation_weighted(ts, popt_snr, N=1e3, border=0.02):
    assert 0 <= border <= 1

    N = int(N)
    out = np.zeros((0,3))
    while out.shape[0] < N:
        amp = 10**np.random.uniform(np.log10(cs.ampgrid[0]),
                                    np.log10(cs.ampgrid[1]), N)
        Prot = 10**np.random.uniform(np.log10(cs.Protgrid[0]),
                                     np.log10(cs.Protgrid[1]), N)
        Gamma = 10**np.random.uniform(0, 1, N)

        # get each planet's snr and corresponding sensitivity
        snr = misc.estimate_Prot_snr(ts, amp, Prot)
        sens = _gammaCDF(snr, *popt_snr)
        accept = (sens > border) & (sens < 1-border)
        out = np.vstack([out, np.array([amp,Prot,Gamma]).T[accept]])
        
    # return only the requested number of parameters
    amp, Prot, Gamma = out[:N].T
    return amp, Prot, Gamma



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

    

def inject_rotation(injrec, ts, amp, Prot, Gamma=np.nan, seed=np.nan):
    '''
    Sample rotation signal from a periodic GP prior.
    '''
    # get transit model
    if np.isfinite(seed):
        np.random.seed(int(seed))
    args = ts.lc.bjd, amp, Prot, Gamma
    injrec.bjd = np.copy(args[0])
    injrec.injected_model = rotation_model(*args)
    snr = misc.estimate_Prot_snr(ts, amp, Prot)
    injrec.argsinjected = np.append(np.copy(args[1:]), snr)

    # inject rotation
    injrec.finjected = injrec.fclean + injrec.injected_model



def rotation_model(bjd, amp, Prot, Gamma, N=1e3):
    # get time stamps instead of sampling all TESS points
    N = int(N)
    t = np.append(np.append(bjd.min(), np.random.choice(bjd, N-2)), bjd.max())
    t = np.sort(t) - t.min()
    
    # sample the flux from the GP prior
    gp = george.GP(george.kernels.ExpSine2Kernel(Gamma, np.log(Prot)))
    ##gp.compute(t, np.ones(bjd.size))
    flux = gp.sample(t)
    flux -= np.nanmin(flux)
    flux *= amp*1e-3 / np.nanmax(flux) # ppt to normalized flux
    flux -= np.nanmedian(flux)
    
    # interpolate to TESS sampling
    fint = interp1d(t, flux)
    return fint(bjd - bjd.min())



def run_gls(injrec, ts, Prot_inj):
    '''
    Compute the GLS periodogram of the binned light curve and look for prominent
    peaks to assess the presence of stellar rotation.
    '''
    # compute gls
    x, y = misc.bin_lc(injrec.bjd, injrec.finjected)
    g = y != 0
    T = 27 * np.max([len(s) for s in ts.lc.sect_ranges])
    gls = Gls((x[g], y[g], np.ones(g.sum())), fend=2, fbeg=1/T)
    periods, power = 1/gls.freq, gls.power

    # check if there's a strong peak indicative of Prot
    gls_pwr = power.max()
    Prot_rec = periods[np.argmax(power)] if gls_pwr >= cs.minGlspwr else np.nan
    is_detected = is_rotation_detected(Prot_inj, Prot_rec)

    return is_detected, gls_pwr



def is_rotation_detected(Prot_inj, Prot_rec, rtol=.02):
    '''
    Given the period of the injected rotation signal, check if it is seen in 
    the gls of the light curve.
    '''    
    # get possible peaks to check
    Ppeaks = []
    for j in range(1,7):
        Ppeaks.append(Prot_inj*j)
        Ppeaks.append(Prot_inj/j)
    Ppeaks = np.sort(Ppeaks)

    # check if the GLS peak is close to the injected peak or a harmonic
    is_detected = np.any((np.isclose(Ppeaks, Prot_rec, rtol=rtol)))
    
    return is_detected

