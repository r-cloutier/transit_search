import numpy as np
import pylab as plt
import get_tess_data as gtd
import median_detrend as mdt 
import transitleastsquares as tls

global t0, plot_dir
t0 = 2457e3
plot_dir = '/Users/ryancloutier/Research/TLS/plots'


def read_in_lightcurve(tic, minsector=1, maxsector=56, use_20sec=False,
                       pltt=True):
    # read in TESS LC
    kwargs = {'minsector': minsector,
              'maxsector': maxsector,
              'use_20sec': use_20sec}
    p = gtd.read_TESS_data(tic, **kwargs)
    bjd, fnorm, efnorm, sectors, qual_flags, texps = p
    Nsect = np.unique(sectors)
    
    # save LC plot
    if pltt:
        plt.figure(figsize=(8,Nsect))
        for i,s in enumerate(np.unique(sectors)):
            g = sectors == s
            plt.subplot(Nsect,1,i+1)
            plt.plot(bjd[g]-t0, fnrom[g], '-', lw=.2, label='sector %i'%s)
            plt.xlabel('BJD - 2,457,000', fontsize=12)
            plt.ylabel('Normalized flux', fontsize=12)
            plt.legend(fontsize=12)
            plt.savefig('%s/rawLC_TIC%i.png'%(plot_dir, tic))
            plt.close('all')
            
    return bjd, fnorm, efnorm, sectors, qual_flags, texps, Nsect



def detrend_lightcurve_median(*args, window_length_hrs=12):
    assert len(args) == 5
    bjd, fnorm, efnorm, sectors, qual_flags = args
    
    # detrend each sector individually and construct outlier mask
    kwargs = {'window_length_days': window_length_hrs/24}
    fdetrend, mask = mdt.detrend_all_sectors(bjd, fnorm, sectors, **kwargs)

    # mask outliers
    p = np.vstack([bjd,fnorm,fdetrend,efnorm,sectors,qual_flags]).T[mask].T

    return p


