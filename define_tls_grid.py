import numpy as np
import pylab as plt
import constants as cs


def get_Ntransit_vs_period(tic, bjd, sectors, pltt=True):
    assert bjd.size == sectors.size
    
    # try getting periods recovered for a given lc
    Pmax = np.arange(6,51)
    N = 500
    Nsect = np.unique(sectors).size
    Ntransits_covered = np.zeros((Nsect,len(Pmax),N))

    # for each sector
    Pmax_per_sector = np.zeros(Nsect)
    for si,s in enumerate(np.unique(sectors)):
        g = sectors == s

        # for each possible maximum period
        for i,p in enumerate(Pmax):
            
            # for random phases
            phis = np.random.uniform(bjd[g].min(), bjd[g].min()+Pmax.max(), N)
            for j,t0 in enumerate(phis):

                # how many mid-transits are covered by the LC?
                Ntransits_covered[si,i,j] = np.sum(abs(_foldAt(bjd[g],p,t0)) < .5*2/60/24/p)

        # get Pmax for this sector
        Pmax_per_sector[si] = Pmax[np.median(Ntransits_covered[si],1) >= 2][-1]


    # plot
    if pltt:
        Nsect = np.unique(sectors).size
        plt.figure(figsize=(8,4))
        for i,s in enumerate(np.unique(sectors)):
            g = sectors == s
            plt.plot(Pmax, np.median(Ntransits_covered[i], axis=1), '-',
                     label='sector %i'%s)
        plt.title('TIC %i'%tic, fontsize=12)
        plt.ylabel('Median # of transits covered', fontsize=12)
        plt.xlabel('Maximum period [days]', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(color='k', lw=.2, ls='--')
        plt.savefig('%s/plots/Ntransits_TIC%i.png'%(cs.repo_dir, tic))
        plt.close('all')

    return Pmax_per_sector, Pmax, Ntransits_covered



                
def _foldAt(t, P, T0=0):
    phi = ((t-T0) % P) / P
    phi[phi>.5] -= 1   # change range from [0,1] to [-0.5,0.5]
    return phi
