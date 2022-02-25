import numpy as np
import pylab as plt
import pdb, misc, os
import constants as cs
from tls_object import transit_search, loadpickle
from gls import Gls
import get_tess_data as gtd
import median_detrend as mdt
import GPexoplanet as gpx
import define_tls_grid as dtg
import transitleastsquares as tls
import planet_vetting as pv


def run_full_planet_search(tic, use_20sec=False, overwrite=False):
    '''
    Run each step of the transit search and save the results.
    '''
    # skip if this TIC already has a completed planet search
    if (is_already_done(tic)) & (overwrite == False):
        return None

    kwargs = {'minsector': cs.minsector, 'maxsector': cs.maxsector,
              'use_20sec': bool(use_20sec), 'pltt': True}
    ts = read_in_lightcurve(tic, **kwargs)

    try:
        detrend_lightcurve_GP(ts)
        ts.lc.GPused = True
    except:
        kwargs = {'window_length_hrs': 12}
        detrend_lightcurve_median(ts, **kwargs)
        ts.lc.GPused = False

    run_tls_Nplanets(ts)
    ts.pickleobject()
    vet_planets(ts)
    ts.DONE = True
    ts.pickleobject()
 
    return ts
    


def is_already_done(tic):
    fname = '%s/MAST/TESS/TIC%i/TESSLC_planetsearch'%(cs.repo_dir, tic)
    return loadpickle(fname).DONE if os.path.exists(fname) else False



def read_in_lightcurve(tic, minsector=1, maxsector=56, use_20sec=False, pltt=True):
    '''
    Download and save available TESS light curves for a specified TIC source.
    '''
    print('\nReading in light curve for TIC %i\n'%tic)

    # read in TESS LC
    kwargs = {'minsector': minsector,
              'maxsector': maxsector,
              'use_20sec': use_20sec}
    p = gtd.read_TESS_data(tic, **kwargs)

    # save light curve
    ts = transit_search(tic, 'TESSLC_planetsearch')
    ts.lc.bjd_raw, ts.lc.fnorm_raw, ts.lc.efnorm_raw, ts.lc.sectors_raw, ts.lc.qual_flags_raw, ts.lc.texps_raw = p[:-3]
    ts.star.Tmag, ts.star.RA, ts.star.Dec = p[-3:]

    # get sectors
    ts.lc.sect_ranges = misc.get_consecutive_sectors(np.unique(ts.lc.sectors_raw))
    ts.lc.Nsect = len(ts.lc.sect_ranges)
    
    # save LC plot
    if pltt:
        plt.figure(figsize=(8,ts.lc.Nsect*4))
        for i,s in enumerate(ts.lc.sect_ranges):
            g = np.in1d(ts.lc.sectors_raw, s)
            plt.subplot(ts.lc.Nsect,1,i+1)
            slabel = '%i'%s[0] if len(s) == 1 else '%i-%i'%(min(s),max(s))
            plt.plot(ts.lc.bjd_raw[g]-cs.t0, ts.lc.fnorm_raw[g], 'o', ms=1,
                     alpha=.5, label='sector %s'%slabel)
            if i == 0: plt.title('TIC %i'%tic, fontsize=12)
            plt.ylabel('Normalized flux', fontsize=12)
            plt.legend(frameon=True, fontsize=12)
        plt.xlabel('BJD - 2,457,000', fontsize=12)
        plt.savefig('%s/MAST/TESS/TIC%i/rawLC.png'%(cs.repo_dir, ts.tic))
        plt.close('all')
            
    return ts




def detrend_lightcurve_median(ts, window_length_hrs=12, pltt=True):
    '''
    Detrend the light curve using a running median with a specified lengthscale.
    '''
    print('\nMedian detrending light curve for TIC %i\n'%ts.tic)

    # detrend each sector individually and construct outlier mask
    kwargs = {'window_length_hrs': window_length_hrs}
    ts.lc.fdetrend_full, ts.lc.detrend_model_full, ts.lc.mask = mdt.detrend_all_sectors(ts.lc.bjd_raw, ts.lc.fnorm_raw, ts.lc.sectors_raw, **kwargs)

    # mask outliers
    p = np.vstack([ts.lc.bjd_raw,ts.lc.fnorm_raw,ts.lc.fdetrend_full,ts.lc.efnorm_raw,ts.lc.sectors_raw,ts.lc.qual_flags_raw,ts.lc.detrend_model_full]).T[ts.lc.mask].T
    ts.lc.bjd, ts.lc.fnorm, ts.lc.fdetrend, ts.lc.efnorm, ts.lc.sectors, ts.lc.qual_flags, ts.lc.detrend_model = p
    ts.lc.efnorm_rescaled = np.zeros_like(ts.lc.efnorm) + np.std(ts.lc.fdetrend)

    # save LC plot
    if pltt:
        plt.figure(figsize=(8,ts.lc.Nsect*4))
        for i,s in enumerate(ts.lc.sect_ranges):
            g = np.in1d(ts.lc.sectors, s)
            plt.subplot(ts.lc.Nsect,1,i+1)
            slabel = '%i'%s[0] if len(s) == 1 else '%i-%i'%(min(s),max(s))
            dy = ts.lc.fdetrend[g].max()-ts.lc.fnorm[g].min()
            plt.plot(ts.lc.bjd[g]-cs.t0, ts.lc.fnorm[g]+dy,
                     'o', ms=1, alpha=.5, label='sector %s'%slabel)
            plt.plot(ts.lc.bjd[g]-cs.t0, ts.lc.fdetrend[g], 'o', ms=1, alpha=.5)
            plt.plot(ts.lc.bjd[g]-cs.t0, ts.lc.detrend_model[g]+dy, 'k-')
            if i == 0: plt.title('TIC %i'%ts.tic, fontsize=12)
            plt.ylabel('Normalized flux', fontsize=12)
            plt.legend(frameon=True, fontsize=12)
        plt.xlabel('BJD - 2,457,000', fontsize=12)
        plt.savefig('%s/MAST/TESS/TIC%i/detrendedLC.png'%(cs.repo_dir, ts.tic))
        plt.close('all')
    


def detrend_lightcurve_GP(ts, pltt=True):
    '''
    Detrend the light curve using a GP with a SHO kernel.
    '''
    print('\nGP detrending light curve for TIC %i\n'%ts.tic)

    # check for rotation
    get_Prot_from_GLS(ts)
    assert np.isfinite(ts.star.Prot)

    # detrend each sector individually and construct outlier mask
    ts.lc.fdetrend_full,ts.lc.mask,ts.lc.map_soln,extras = gpx.detrend_light_curve(ts.lc.bjd_raw, ts.lc.fnorm_raw, 
                                                                                   ts.lc.efnorm_raw, ts.lc.sectors_raw, 
                                                                                   ts.star.Prot)

    # mask outliers
    p = np.vstack([ts.lc.bjd_raw,ts.lc.fnorm_raw,ts.lc.fdetrend_full,ts.lc.efnorm_raw,ts.lc.sectors_raw,ts.lc.qual_flags_raw]).T[ts.lc.mask].T
    ts.lc.bjd, ts.lc.fnorm, ts.lc.fdetrend, ts.lc.efnorm, ts.lc.sectors, ts.lc.qual_flags = p
    ts.lc.efnorm_rescaled = np.zeros_like(ts.lc.efnorm) + np.std(ts.lc.fdetrend)
    ts.lc.detrend_model = ts.lc.fnorm / ts.lc.fdetrend

    # save LC plot
    if pltt:
        plt.figure(figsize=(8,ts.lc.Nsect*4))
        for i,s in enumerate(ts.lc.sect_ranges):
            g = np.in1d(ts.lc.sectors, s)
            plt.subplot(ts.lc.Nsect,1,i+1)
            slabel = '%i'%s[0] if len(s) == 1 else '%i-%i'%(min(s),max(s))
            dy = (ts.lc.fdetrend[g].max()-ts.lc.fnorm[g].min())
            plt.plot(ts.lc.bjd[g]-cs.t0, ts.lc.fnorm[g]+dy,
                     'o', ms=1, alpha=.5, label='sector %s'%slabel)
            plt.plot(ts.lc.bjd[g]-cs.t0, ts.lc.fdetrend[g], 'o', ms=1, alpha=.5)
            plt.plot(ts.lc.bjd[g]-cs.t0, ts.lc.detrend_model[g]+dy, 'k-')
            if i == 0: plt.title('TIC %i'%ts.tic, fontsize=12)
            plt.ylabel('Normalized flux', fontsize=12)
            plt.legend(frameon=True, fontsize=12)
        plt.xlabel('BJD - 2,457,000', fontsize=12)
        plt.savefig('%s/MAST/TESS/TIC%i/detrendedLC.png'%(cs.repo_dir, ts.tic))
        plt.close('all')



def get_Prot_from_GLS(ts, pltt=True):
    '''
    Compute the GLS periodogram of the binned light curve and look for prominent 
    peaks to assess the presence of stellar rotation.
    '''
    # compute gls
    x, y = misc.bin_lc(ts.lc.bjd_raw, ts.lc.fnorm_raw)
    g = y != 0
    T = 27 * np.max([len(s) for s in ts.lc.sect_ranges])
    gls = Gls((x[g], y[g], np.ones(g.sum())), fend=2, fbeg=1/T)

    # save stuff
    ts.gls.Pmin, ts.gls.Pmax = 1/gls.fend, 1/gls.fbeg
    ts.gls.periods, ts.gls.power = 1/gls.freq, gls.power
 
    # check if there's a strong peak indicative of Prot
    ts.star.Prot = ts.gls.periods[np.argmax(ts.gls.power)] if ts.gls.power.max() >= .4 else np.nan
 
    # save gls plot
    if pltt:
        plt.figure(figsize=(8,4))
        plt.plot(ts.gls.periods, ts.gls.power, 'k-', lw=.9, zorder=2)
        plt.axvline(ts.star.Prot, ls='--', lw=2, zorder=1)
        plt.xscale('log')
        plt.title('Prot = %.3f days'%ts.star.Prot, fontsize=12)
        plt.ylabel('GLS power', fontsize=12), plt.xlabel('Period [days]', fontsize=12)
        plt.savefig('%s/MAST/TESS/TIC%i/gls.png'%(cs.repo_dir, ts.tic))
        plt.close('all') 



def run_tls_Nplanets(ts, pltt=True):
    '''
    Run the Transit-Least-Squares search for multiple signals on an input 
    (detrended) light curve.
    '''
    # get approximate stellar parameters
    p = np.ascontiguousarray(tls.catalog_info(TIC_ID=ts.tic))
    ts.star.ab = p[0]
    ts.star.Ms, ts.star.Ms_min, ts.star.Ms_max, ts.star.Rs, ts.star.Rs_min, ts.star.Rs_max, ts.star.Teff = p[1:].astype(float)
    ts.star.reliable_Ms, ts.star.reliable_Rs, ts.star.reliable_Teff = np.repeat(True,3)

    # check that the stellar parameters are known, otherwise add a placeholder
    if np.any(np.isnan([ts.star.Ms, ts.star.Ms_max, ts.star.Ms_min])):
        ts.star.reliable_Ms = False
        ts.star.Ms, ts.star.Ms_max, ts.star.Ms_min = .3, .03, .03

    if np.any(np.isnan([ts.star.Rs, ts.star.Rs_max, ts.star.Rs_min])):
        ts.star.reliable_Rs = False
        ts.star.Rs, ts.star.Rs_max, ts.star.Rs_min = .3, .03, .03

    if np.isnan(ts.star.Teff):
        ts.star.reliable_Teff = False
        ts.star.Teff = 3300

    # plot Ntransits vs period
    _=dtg.get_Ntransit_vs_period(ts.tic, ts.lc.bjd, ts.lc.sectors)

    for i,s in enumerate(ts.lc.sect_ranges):
        
        g = np.in1d(ts.lc.sectors, s)
        lc_input = ts.lc.bjd[g], ts.lc.fdetrend[g], ts.lc.efnorm_rescaled[g]
        slabel = '%i'%s[0] if len(s) == 1 else '%i-%i'%(min(s),max(s))

        # get maximum period for 2 transits on average
        Pmax,_,_ = dtg.get_Ntransit_vs_period(ts.tic, ts.lc.bjd[g], ts.lc.sectors[g], pltt=False)

        # run tls on this sector
        iter_count = 1
        results = _run_tls(*lc_input, ts.star.ab, period_max=float(Pmax))
        results.snr = misc.estimate_snr(ts, results['period'], results['T0'], misc.m2Rearth(misc.Rsun2m(results['rp_rs']*ts.star.Rs))) if np.isfinite(results['period']) else np.nan
        setattr(ts.tls, 'results_%i_s%s'%(iter_count,slabel), results)

        # mask out found signal
        lc_input = _mask_transits(*lc_input, getattr(ts.tls, 'results_%i_s%s'%(iter_count,slabel)))

        while (iter_count <= cs.Nplanets_max) and (np.any(results.power) >= cs.SDEthreshold):

            print('\nRunning TLS for planet %i (sector(s) %s)\n'%(iter_count,slabel))
            
            # run tls on this sector
            iter_count += 1
            results = _run_tls(*lc_input, ts.star.ab, period_max=float(Pmax))
            setattr(ts.tls, 'results_%i_s%s'%(iter_count,slabel), results)

            # mask out found signal
            lc_input = _mask_transits(*lc_input, getattr(ts.tls, 'results_%i_s%s'%(iter_count,slabel)))

            if pltt:
                # plotting
                plt.figure(figsize=(8,4))
                plt.title('TIC %i - Sector(s) %s - TLS Run %i - Period %.3f days'%(ts.tic,slabel,iter_count,results.period))
                plt.plot(results.periods, results.power, 'k-', lw=.5)
                plt.axvline(results['period'], alpha=.4, lw=3)
                plt.xlim(np.min(results['periods']), np.max(results['periods']))
                plt.axvline(results.period, ls='--', alpha=.4)
                for j in range(2,10):
                    plt.axvline(j*results.period, ls='--', alpha=.4)
                    plt.axvline(results.period/j, ls='--', alpha=.4)
                plt.ylabel('SDE', fontsize=12)
                plt.xlabel('Period [days]', fontsize=12)
                plt.xlim(0, np.max(results.periods)*1.02)
                plt.savefig('%s/MAST/TESS/TIC%i/sde_s%s_run%i'%(cs.repo_dir,ts.tic,slabel,iter_count))
                plt.close('all')




def vet_planets(ts):
    '''
    Given the TLS results, vet planets using various metrics. 
    '''
    print('\nVetting planet candidates around TIC %i\n'%ts.tic)
    pv.get_POIs(ts)
    pv.vet_SDE(ts)
    pv.vet_snr(ts)
    pv.vet_multiple_sectors(ts)
    pv.vet_odd_even_difference(ts)
    pv.vet_Prot(ts)
    #pv.plot_light_curves(ts)
    pv.save_planet_parameters(ts)




def _run_tls(bjd, fdetrend, efnorm, ab, period_max=0):
    model = tls.transitleastsquares(bjd, fdetrend, efnorm)
    if period_max > 0: 
        results = model.power(u=ab, period_max=period_max, 
                              period_min=np.min(cs.Pgrid))
    else:
        results = model.power(u=ab, period_min=np.min(cs.Pgrid[0]))
    return results



def _mask_transits(bjd, fdetrend, efnorm, results):
    '''
    Mask in-transit points so that the TLS can be rerun on the masked LC.
    '''
    intransit = tls.transit_mask(bjd, results['period'], 2*results['duration'], results['T0'])
    fdetrend_2 = fdetrend[~intransit]
    efnorm_2 = efnorm[~intransit]
    bjd_2 = bjd[~intransit]
    bjd_2, fdetrend_2, efnorm_2 = tls.cleaned_array(bjd_2, fdetrend_2, efnorm_2)
    return bjd_2, fdetrend_2, efnorm_2
