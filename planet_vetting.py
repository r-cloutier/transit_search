import numpy as np
import pylab as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import pandas as pd
import constants as cs
import misc, pdb

MAD = lambda arr: np.nanmedian(abs(arr - np.nanmedian(arr)))

def get_POIs(ts, injrec=False):
    '''
    Given a set of tls result dictionaries (i.e. from multiple TLS runs and/or
    multiple sectors), consolidate periods of interest. 

    Behaviour is slightly different between the planet search and vetting of 
    injected planets.
    '''
    # planet search for injrec?
    tlsobj = ts.injrec.tls if injrec else ts.tls
    
    # collect all periods
    POIsv1 = np.zeros((0,11)) 
    for res in tlsobj.__dict__.keys():
        POIsv1 = np.vstack([POIsv1, [getattr(tlsobj,res).period,
                                     getattr(tlsobj,res).T0,
                                     getattr(tlsobj,res).duration*24,
                                     1e3*(1-getattr(tlsobj,res).depth),
                                     getattr(tlsobj,res).rp_rs,
                                     getattr(tlsobj,res).chi2_min,
                                     getattr(tlsobj,res).chi2red_min,
                                     getattr(tlsobj,res).SDE_raw,
                                     getattr(tlsobj,res).SDE,
                                     getattr(tlsobj,res).snr,
                                     getattr(tlsobj,res).odd_even_mismatch]])
    POIsv1 = POIsv1[np.argsort(POIsv1,0)[:,0]]
    
    # identify duplicates
    POIsv2 = np.zeros((0,13))
    for p in POIsv1[:,0]:
        if np.isnan(p):
            continue
        duplicates = np.isclose(POIsv1[:,0], p, rtol=cs.P_duplicate_fraction)
        Nocc = np.sum(duplicates)

        # combine period from multiple sectors if available
        if Nocc > 1:
            N = np.round(abs(np.diff(POIsv1[duplicates,1])) / POIsv1[duplicates,0][:-1])
            avgP = np.average(abs(np.diff(POIsv1[duplicates,1])) / N, weights=POIsv1[duplicates,8][:-1])
        else:
            avgP = np.average(POIsv1[duplicates,0], weights=POIsv1[duplicates,8])

        avgT0 = POIsv1[duplicates,1][np.argmax(POIsv1[duplicates,8])]   # cannot average T0s from multiple sectors
        avgD = np.average(POIsv1[duplicates,2], weights=POIsv1[duplicates,8])
        avgZ = np.average(POIsv1[duplicates,3], weights=POIsv1[duplicates,8])
        avgrpRs = np.average(POIsv1[duplicates,4], weights=POIsv1[duplicates,8])
        avgRp = misc.m2Rearth(misc.Rsun2m(avgrpRs * ts.star.Rs))
        avgchi2min = np.average(POIsv1[duplicates,5], weights=POIsv1[duplicates,8])
        avgchi2redmin = np.average(POIsv1[duplicates,6], weights=POIsv1[duplicates,8])
        avgSDEraw = np.average(POIsv1[duplicates,7], weights=POIsv1[duplicates,8])
        avgSDE = np.average(POIsv1[duplicates,8], weights=POIsv1[duplicates,8])
        avgsnr = np.average(POIsv1[duplicates,9], weights=POIsv1[duplicates,8])
        avgOED = np.average(POIsv1[duplicates,10], weights=POIsv1[duplicates,8])
        POIsv2 = np.vstack([POIsv2, [avgP,avgT0,avgD,avgZ,avgrpRs,avgRp,avgchi2min,avgchi2redmin,avgSDEraw,avgSDE,avgsnr,avgOED,Nocc]])

    # remove duplicates
    _,inds = np.unique(POIsv2[:,0], return_index=True)
    POIsv3 = POIsv2[inds]
    
    # leave multiples alone because if they surive the iterative TLS,
    # they may be real (e.g. L 98-59)
    # save parameters for each OI
    vetobj = ts.injrec.vetting if injrec else ts.vetting
    for i,s in enumerate(['POIs','T0OIs','DOIs','ZOIs','rpRsOIs','RpOIs','chi2minOIs','chi2redminOIs','SDErawOIs','SDEOIs','snrOIs','oddevendiff_sigma','NoccurrencesOIs']):
        setattr(vetobj, s, POIsv3[:,i])

    vetobj.vetting_mask = np.ones(POIsv3.shape[0]).astype(bool)
    vetobj.conditions = np.zeros(POIsv3.shape[0])

    

def vet_SDE(ts, injrec=False):
    vetobj = ts.injrec.vetting if injrec else ts.vetting
    g = vetobj.SDEOIs >= cs.SDEthreshold
    vetobj.vetting_mask *= g
    vetobj.conditions[np.invert(g)] += 1  # condition1


def vet_snr(ts, injrec=False):
    vetobj = ts.injrec.vetting if injrec else ts.vetting
    g = vetobj.snrOIs >= cs.SNRthreshold
    vetobj.vetting_mask *= g
    vetobj.conditions[np.invert(g)] += 2 # condition2


def vet_multiple_sectors(ts, injrec=False):
    '''
    If multiple sectors have been observed, vet planets that were not recovered
    in multiple sectors. 
    '''
    vetobj = ts.injrec.vetting if injrec else ts.vetting
    if ts.lc.Nsect <= 1:
        pass
    else:
        g = vetobj.NoccurrencesOIs > 1
        vetobj.vetting_mask *= g
        vetobj.conditions[np.invert(g)] += 4  # condition4

        
def vet_odd_even_difference(ts, injrec=False):
    '''
    Check for a significant odd-even difference in the transit depths.
    '''
    vetobj = ts.injrec.vetting if injrec else ts.vetting
    g = vetobj.oddevendiff_sigma < 3
    vetobj.vetting_mask *= g
    vetobj.conditions[np.invert(g)] += 8  # condition8


def vet_Prot(ts, rtol=0.02, injrec=False):
    '''
    Check that the planet candidate is not close to Prot or a harmonic.
    '''
    vetobj = ts.injrec.vetting if injrec else ts.vetting
    # get rotation periods and harmonics to reject
    Prots = []
    for j in range(1,4):
        Prots.append(ts.star.Prot_gls/j)
    Prots = np.sort(Prots)

    # check each POI
    for i,p in enumerate(vetobj.POIs):
        g = np.all(np.invert(np.isclose(Prots, p, rtol=rtol)))
        vetobj.vetting_mask[i] *= g
        if not g: vetobj.conditions[i] += 16  # condition16



def vet_tls_Prot(ts, rtol=0.02, sig=3, injrec=False):
    '''
    Check that the tls did not likely return a rotation signature, which can happen if the gls 
    fails to flag stellar rotation.
    '''
    vetobj = ts.injrec.vetting if injrec else ts.vetting
    tlsobj = ts.injrec.tls if injrec else ts.tls

    for i,p in enumerate(vetobj.POIs):
        rotation_signature = []
        for k in tlsobj.__dict__.keys():

            # check if sinusoidal model is favoured over a transit interpretation
            theta = 1-getattr(tlsobj,k)['depth'], .5, 1, 1
            x, y, ey = getattr(tlsobj,k)['folded_phase'], getattr(tlsobj,k)['folded_y'], np.repeat(np.nanmedian(ts.lc.efnorm_rescaled),getattr(tlsobj,k)['folded_y'].size)
            popt,_ = curve_fit(misc.sinemodel, x, y, p0=theta, bounds=((theta[0],0,0,.9),(1,1,np.inf,1.1)))
            model = misc.sinemodel(x, *popt)
            _,_,dBIC = misc.DeltaBIC(y, ey, model, getattr(tlsobj,k)['model_folded_model'], k=4, knull=5)
            dBIC_favours_rotation = dBIC <= -10

            # check if TLS peaks look like rotation+harmonics
            per, pwr = getattr(tlsobj,k).periods, getattr(tlsobj,k).power
            is_harmonic_significant = np.zeros(4).astype(bool)    # check that sde at the harmonics are large
            is_harmonic_decreasing = np.zeros(3).astype(bool)     # check that the sde of the peaks are decreasing
            for n in range(1,5):
                g = np.isclose(per, p*n, rtol=rtol) 
                if g.sum() > 0:
                    is_harmonic_significant[n-1] = np.max(pwr[g]) >= sig*MAD(pwr)
                    if n > 1: is_harmonic_decreasing[n-2] = np.max(pwr[g]) <= np.max(pwr[np.isclose(per, p*(n-1), rtol=rtol)])
                else:
                    is_harmonic_significant[n-1] = False
            # false POI if all the harmonics have significant peaks and show decreasing pwr with increasing period
            g = np.all(is_harmonic_significant) and np.all(is_harmonic_decreasing) and (dBIC_favours_rotation)
            rotation_signature.append(g)

        vetobj.vetting_mask[i] *= np.sum(rotation_signature)/len(rotation_signature) <= .5  # need at least half of sectors to favour rotation
        
        # save the tls-recovered period if found
        if np.sum(rotation_signature)/len(rotation_signature) >= .5: 
            vetobj.conditions[i] += 32  # condition32
            ts.star.Prot_tls = p
        elif not hasattr(ts.star,'Prot_tls'):
            ts.star.Prot_tls = np.nan


def model_comparison_deprecated(ts, injrec=False):
    '''
    In each sector that each PC is found in, check that the delta BIC favours 
    the transit model over the null hypothesis (i.e. a flat line).
    '''
    vetobj = ts.injrec.vetting if injrec else ts.vetting
    tlsobj = ts.injrec.tls if injrec else ts.tls

    for i,p in enumerate(vetobj.POIs):
        dBIC_vetted = []  # if True, then transit model is favoured by the dBIC
        for k in tlsobj.__dict__.keys():
            # get tls results
            res = getattr(tlsobj, k)

            if np.isclose(res['period'], p, rtol=cs.P_duplicate_fraction):
                # interpolate transit model grid to observation epochs 
                # (add zero and one to the phase edges to avoid bound_errors)
                fint = interp1d(np.hstack([0,res['model_folded_phase'],1]), np.hstack([1,res['model_folded_model'],1]))
                model = fint(res['folded_phase'])

                # compute delta BIC (i.e. transit minus line)
                ey = np.median(ts.lc.efnorm_rescaled)
                BIC_transit, BIC_null, dBIC = misc.DeltaBIC(res['folded_y'], ey, model, np.ones_like(model))
                dBIC_vetted.append(dBIC <= -10)

        # does each sector favour this PC's transit model?
        g = np.all(dBIC_vetted)
        vetobj.vetting_mask[i] *= g
        if not g: vetobj.conditions[i] += 64  # condition64

        

def model_comparison(ts, injrec=False):
    '''
    For each PC, check that the delta BIC favours the transit model over the
    null hypothesis (i.e. a flat line) when computed for all sectors, not just 
    on each set of consecutive sectors. 

    The former generates better agreement with TOIs with many examples.
    '''
    vetobj = ts.injrec.vetting if injrec else ts.vetting
    tlsobj = ts.injrec.tls if injrec else ts.tls
    
    for i,p in enumerate(vetobj.POIs):
        
        y, ey, model = np.zeros(0), np.zeros(0), np.zeros(0)
        for k in tlsobj.__dict__.keys():
            # get tls results
            res = getattr(tlsobj, k)

            if np.isclose(res['period'], p, rtol=cs.P_duplicate_fraction):
                # interpolate transit model grid to observation epochs 
                # (add zero and one to the phase edges to avoid bound_errors)
                fint = interp1d(np.hstack([0,res['model_folded_phase'],1]), np.hstack([1,res['model_folded_model'],1]))
                model = np.append(model, fint(res['folded_phase']))

                y = np.append(y, res['folded_y'])

                ey = np.append(ey, np.repeat(np.nanmedian(ts.lc.efnorm_rescaled), res['folded_y'].size))

        # compute delta BIC for this OI (i.e. transit minus line)
        BIC_transit, BIC_null, dBIC = misc.DeltaBIC(y, ey, model, np.ones_like(model))
        dBIC_vetted = dBIC <= -10

        # does each sector favour this PC's transit model?
        vetobj.vetting_mask[i] *= dBIC_vetted
        if not dBIC_vetted: vetobj.conditions[i] += 64  # condition64

 

def vet_edges(ts, injrec=False):
    vetobj = ts.injrec.vetting if injrec else ts.vetting
    vetobj.intransit_mask = np.zeros((ts.lc.bjd.size, ts.vetting.POIs.size)).astype(bool)  # save intransit mask for each PC
    g = np.zeros_like(vetobj.vetting_mask).astype(bool)
    for i,p in enumerate(vetobj.POIs):
        vetobj.intransit_mask[:,i] = abs(misc.foldAt(ts.lc.bjd, p, vetobj.T0OIs[i])) <= vetobj.DOIs[i]/2/24/p
        intransit = np.where(vetobj.intransit_mask[:,i])[0]
        edges = np.sort(np.hstack([0, np.where(np.diff(ts.lc.bjd) > .1)[0], np.where(np.diff(ts.lc.bjd) > .1)[0]+1, ts.lc.bjd.size-1]))
        g[i] = np.all(np.invert(np.in1d(edges, intransit)))  # all edges indices are not in-transit
    vetobj.vetting_mask *= g
    vetobj.conditions[np.invert(g)] += 128  # condition128



def plot_light_curves(ts):
    '''
    For each vetted PC, plot the light curve to highlight the transits.
    '''
    mask = ts.vetting.vetting_mask
    Np = mask.sum()

    for i in range(Np):

        # get matching transit model
        for k in ts.tls.__dict__.keys():        
            if np.isclose(ts.vetting.POIs[mask][i], getattr(ts.tls,k).period, rtol=cs.P_duplicate_fraction):

                # plot light curve for each sector
                plt.figure(figsize=(8,ts.lc.Nsect*4))
                for j,s in enumerate(ts.lc.sect_ranges):
                    
                    g = np.in1d(ts.lc.sectors, s)
                    slabel = '%i'%s[0] if len(s) == 1 else '%i-%i'%(min(s),max(s))
                    phase = misc.foldAt(ts.lc.bjd[g], ts.vetting.POIs[mask][i], ts.vetting.T0OIs[mask][i])
                    plt.plot(phase*ts.vetting.POIs[mask][i]*24, ts.lc.fdetrend[g], 'o', ms=1, alpha=.5, label='sector %s'%slabel)
                    
                    # plot transit model
                    model_phase = misc.foldAt(getattr(ts.tls,k).model_folded_phase, 1, .5)
                    fint = interp1d(model_phase, getattr(ts.tls,k).model_folded_model)
                    plt.plot(phase*ts.vetting.POIs[mask][i]*24, fint(phase), '-')
                    
                    # customize
                    if j == 0: plt.title('TIC %i'%ts.tic, fontsize=12)
                    plt.xlabel('Time since mid-transit [hours]', fontsize=12)
                    plt.ylabel('Normalized flux', fontsize=12)
                    plt.xlim(())

                # save figure
                #plt.savefig('%s/MAST/TESS/TIC%i/transitmodel_planet%i.png'%(cs.repo_dir, ts.tic, i+1))
                #plt.close('all')
                plt.show()

            # continue to the next planet
            continue
            

def identify_conditions(ts, injrec=False):
    '''Given the base 2 condition flags, identify the individual flags.'''
    vetobj = ts.injrec.vetting if injrec else ts.vetting
    vetobj.conditions_indiv = []

    for i,c in enumerate(vetobj.conditions):
        v = []
        while (c > 0): 
            v.append(int(c % 2)) 
            c = int(c / 2)
        # list of individual flags for each OI
        vetobj.conditions_indiv.append(list(2**np.where(v)[0]))

    # save parameter definitions
    p = np.genfromtxt('/n/home10/rcloutier/TLS/vetting_flags.txt', delimiter=',', dtype='|S70')
    flags, labels = p[:,0].astype(int), p[:,1].astype(str)
    vetobj.conditions_defs = {}
    for i,f in enumerate(flags):
        vetobj.conditions_defs[f] = labels[i]


def save_planet_parameters(ts):
    N = ts.vetting.POIs.size
    outp = np.vstack([np.repeat(ts.tic, N), np.repeat('-'.join(np.unique(ts.lc.sectors).astype(int).astype(str)),N),
                      np.repeat(ts.star.RA, N), np.repeat(ts.star.Dec, N),
                      np.repeat(ts.star.Tmag, N), np.repeat(ts.star.Teff, N), 
                      np.repeat(ts.star.Ms, N), np.repeat(ts.star.Rs, N),
                      ts.vetting.POIs, ts.vetting.T0OIs, ts.vetting.DOIs,
                      ts.vetting.ZOIs, ts.vetting.rpRsOIs, ts.vetting.RpOIs,ts.vetting.chi2minOIs,
                      ts.vetting.chi2redminOIs, ts.vetting.SDErawOIs,
                      ts.vetting.SDEOIs, ts.vetting.snrOIs,
                      ts.vetting.oddevendiff_sigma, ts.vetting.NoccurrencesOIs,
                      np.repeat(ts.lc.Nsect, N), ts.vetting.vetting_mask, ts.vetting.conditions,
                      np.repeat(ts.DONEcheck_version, N)]).T
    df = pd.DataFrame(outp, columns=['TIC','sectors','RA','Dec','Tmag','Teff','Ms','Rs',
                                     'P','T0','duration_hrs','depth_ppt','rpRs','Rp',
                                     'chi2min','chi2redmin','SDEraw','SDE','snr','oddevendiff_sig',
                                     'Noccurrences','Nsectors','vetted?','vetting_conditions','DONEcheck_version'])
    df = df.sort_values('SDE', ascending=False)
    df.to_csv('%s/MAST/TESS/TIC%i/planetparams_%i.csv'%(cs.repo_dir, ts.tic, ts.tic), index=False)
