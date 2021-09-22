import numpy as np
import pylab as plt
import pdb
import constants as cs
from tls_object import transit_search, loadpickle
import get_tess_data as gtd
import median_detrend as mdt
import define_tls_grid as dtg
import transitleastsquares as tls


def run_full_planet_search(tic, use_20sec=False):
    '''
    Run each step of the transit search and save the results.
    '''
    kwargs = {'minsector': cs.minsector,
              'maxsector': cs.maxsector,
              'use_20sec': use_20sec,
              'pltt': True}
    ts = read_in_lightcurve(tic, **kwargs)
    detrend_lightcurve_median(ts, window_length_hrs=12)
    run_tls_Nplanets(ts)

    # need to do some vetting here
    
    return ts
    


def read_in_lightcurve(tic, minsector=1, maxsector=56, use_20sec=False, pltt=True):
    '''
    Download and save available TESS light curves for a specified TIC source.
    '''
    # read in TESS LC
    kwargs = {'minsector': minsector,
              'maxsector': maxsector,
              'use_20sec': use_20sec}
    p = gtd.read_TESS_data(tic, **kwargs)

    # save light curve
    ts = transit_search(tic, 'TESSLC_planetsearch')
    ts.lc.bjd_raw, ts.lc.fnorm_raw, ts.lc.efnorm_raw, ts.lc.sectors_raw, ts.lc.qual_flags_raw, ts.lc.texps_raw = p
    Nsect = np.unique(ts.lc.sectors_raw).size
    
    # save LC plot
    if pltt:
        plt.figure(figsize=(8,Nsect*4))
        for i,s in enumerate(np.unique(ts.lc.sectors_raw)):
            g = ts.lc.sectors_raw == s
            plt.subplot(Nsect,1,i+1)
            plt.plot(ts.lc.bjd_raw[g]-cs.t0, ts.lc.fnorm_raw[g], '-', lw=.2,
                     label='sector %i'%s)
            if i == 0: plt.title('TIC %i'%tic, fontsize=12)
            plt.ylabel('Normalized flux', fontsize=12)
            plt.legend(fontsize=12)
        plt.xlabel('BJD - 2,457,000', fontsize=12)
        plt.savefig('%s/MAST/TESS/TIC%i/rawLC.png'%(cs.repo_dir, ts.tic))
        plt.close('all')
            
    return ts




def detrend_lightcurve_median(ts, window_length_hrs=12, pltt=True):
    '''
    Detrend the light curve using a running median with a specified lengthscale.
    '''
    # detrend each sector individually and construct outlier mask
    kwargs = {'window_length_days': window_length_hrs/24}
    ts.lc.fdetrend_full, ts.lc.mask = mdt.detrend_all_sectors(ts.lc.bjd_raw, ts.lc.fnorm_raw, ts.lc.sectors_raw, **kwargs)

    # mask outliers
    p = np.vstack([ts.lc.bjd_raw,ts.lc.fnorm_raw,ts.lc.fdetrend_full,ts.lc.efnorm_raw,ts.lc.sectors_raw,ts.lc.qual_flags_raw]).T[ts.lc.mask].T
    ts.lc.bjd, ts.lc.fnorm, ts.lc.fdetrend, ts.lc.efnorm, ts.lc.sectors, ts.lc.qual_flags = p
    Nsect = np.unique(ts.lc.sectors).size

    # save LC plot
    if pltt:
        plt.figure(figsize=(8,Nsect*4))
        for i,s in enumerate(np.unique(ts.lc.sectors)):
            g = ts.lc.sectors == s
            plt.subplot(Nsect,1,i+1)
            plt.plot(ts.lc.bjd[g]-cs.t0,
                     ts.lc.fnorm[g]+(ts.lc.fdetrend[g].max()-ts.lc.fnorm[g].min()),
                     '-', lw=.2, label='sector %i'%s)
            plt.plot(ts.lc.bjd[g]-cs.t0, ts.lc.fdetrend[g], '-', lw=.2)
            if i == 0: plt.title('TIC %i'%ts.tic, fontsize=12)
            plt.ylabel('Normalized flux', fontsize=12)
            plt.legend(fontsize=12)
        plt.xlabel('BJD - 2,457,000', fontsize=12)
        plt.savefig('%s/MAST/TESS/TIC%i/detrendedLC.png'%(cs.repo_dir, ts.tic))
        plt.close('all')
    


def run_tls_Nplanets(ts, Nplanets_max=3, pltt=True):
    '''
    Run the Transit-Least-Squares search for multiple signals on an input 
    (detrended) light curve.
    '''
    # get approximate stellar parameters
    p = tls.catalog_info(TIC_ID=ts.tic)
    ts.star.ab, ts.star.Ms, ts.star.Ms_min, ts.star.Ms_max, ts.star.Rs, ts.star.Rs_min, ts.star.Rs_max = p

    # get maximum period for 2 transits on average
    Pmax,_,_ = dtg.get_Ntransit_vs_period(ts.tic, ts.lc.bjd, ts.lc.sectors)
    
    # run the tls iteratively (multiple signals) and on each sector
    for i,s in enumerate(np.unique(ts.lc.sectors)):
        
        g = ts.lc.sectors == s
        lc_input = ts.lc.bjd[g], ts.lc.fdetrend[g], ts.lc.efnorm[g]

        for n in range(Nplanets_max):

            print('Running TLS for planet %i (sector %i)'%(n+1,s))
            
            # run tls on this sector
            results = _run_tls(*lc_input, ts.star.ab, period_max=Pmax[i])
            setattr(ts.tls, 'results_%i_s%.2d'%(n+1,s), results)

            # mask out found signal
            lc_input = _mask_transits(*lc_input, getattr(ts.tls, 'results_%i_s%.2d'%(n+1,s)))

            if pltt:
                # plotting
                plt.figure(figsize=(8,4))
                plt.title('TIC %i - Sector %.2d - TLS Run %i'%(ts.tic,s,n+1))
                plt.plot(results.periods, results.power_raw, 'k-', lw=.5)
                plt.axvline(results['period'], alpha=.4, lw=3)
                plt.xlim(np.min(results['periods']), np.max(results['periods']))
                plt.axvline(results.period, ls='--', alpha=.4)
                for j in range(2,10):
                    plt.axvline(j*results.period, ls='--', alpha=.4)
                    plt.axvline(results.period/j, ls='--', alpha=.4)
                plt.ylabel('SDE_raw', fontsize=12)
                plt.xlabel('Period [days]', fontsize=12)
                plt.xlim(0, np.max(results.periods)*1.02)
                plt.savefig('%s/MAST/TESS/TIC%i/sde_s%.2d_run%i'%(cs.repo_dir,ts.tic,s,n+1))
                plt.close('all')




def _run_tls(bjd, fdetrend, efnorm, ab, period_max=0):
    model = tls.transitleastsquares(bjd, fdetrend, efnorm)
    if period_max > 0: 
        results = model.power(u=ab, period_max=period_max)
    else:
        results = model.power(u=ab)
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
