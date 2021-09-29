import numpy as np
import pylab as plt
import constants as cs
import misc


def get_Ntransit_vs_period(tic, bjd, sectors, pltt=True):
    assert bjd.size == sectors.size
    
    # try getting periods recovered for a given lc
    Pmax = np.arange(6,51)
    N = 500
    sect_ranges = misc.get_consecutive_sectors(np.unique(sectors))
    Nsect = len(sect_ranges)
    Ntransits_covered = np.zeros((Nsect,len(Pmax),N))

    # for each sector
    Pmax_per_sector = np.zeros(Nsect)
    for si,s in enumerate(sect_ranges):
        g = np.in1d(sectors, s)

        # for each possible maximum period
        for i,p in enumerate(Pmax):
            
            # for random phases
            phis = np.random.uniform(bjd[g].min(), bjd[g].min()+Pmax.max(), N)
            for j,t0 in enumerate(phis):

                # how many mid-transits are covered by the LC?
                cadence = np.median(np.diff(bjd))
                Ntransits_covered[si,i,j] = np.sum(abs(_foldAt(bjd[g],p,t0)) < .5*cadence/60/24/p)

        # get Pmax for this sector
        Pmax_per_sector[si] = Pmax[np.median(Ntransits_covered[si],1) >= 2][-1]


    # plot
    if pltt:
        plt.figure(figsize=(8,4))
        for si,s in enumerate(sect_ranges):
            g = np.in1d(sectors, s)
            slabel = '%i'%s[0] if len(s) == 1 else '%i-%i'%(min(s),max(s))
            plt.plot(Pmax, np.median(Ntransits_covered[si], axis=1), '-',
                     label='sector %s'%slabel)
            plt.axvline(Pmax_per_sector[si], ls='--', lw=.8, color='k')
        plt.title('TIC %i'%tic, fontsize=12)
        plt.ylabel('Median # of transits covered', fontsize=12)
        plt.xlabel('Maximum period [days]', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(color='k', lw=.2, ls='--')
        plt.savefig('%s/MAST/TESS/TIC%i/Ntransits.png'%(cs.repo_dir, tic))
        plt.close('all')

    return Pmax_per_sector, Pmax, Ntransits_covered



                
def _foldAt(t, P, T0=0):
    phi = ((t-T0) % P) / P
    phi[phi>.5] -= 1   # change range from [0,1] to [-0.5,0.5]
    return phi
