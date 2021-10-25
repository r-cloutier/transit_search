import numpy as np
import pandas as pd
import constants as cs


def get_POIs(ts):
    '''
    Given a set of tls result dictionaries (i.e. from multiple TLS runs and/or
    multiple sectors), consolidate periods of interest. 
    '''
    # collect all periods
    POIsv1 = np.zeros((0,11)) 
    for res in ts.tls.__dict__.keys():
        POIsv1 = np.vstack([POIsv1, [getattr(ts.tls,res).period,
                                     getattr(ts.tls,res).T0,
                                     getattr(ts.tls,res).duration*24,
                                     1e3*(1-getattr(ts.tls,res).depth),
                                     getattr(ts.tls,res).rp_rs,
                                     getattr(ts.tls,res).chi2_min,
                                     getattr(ts.tls,res).chi2red_min,
                                     getattr(ts.tls,res).SDE_raw,
                                     getattr(ts.tls,res).SDE,
                                     getattr(ts.tls,res).snr,
                                     getattr(ts.tls,res).odd_even_mismatch]])
    POIsv1 = POIsv1[np.argsort(POIsv1,0)[:,0]]
    
    # identify duplicates
    POIsv2 = np.zeros((0,12))
    for p in POIsv1[:,0]:
        duplicates = np.isclose(POIsv1[:,0], p, rtol=cs.P_duplicate_fraction)
        avgP = np.average(POIsv1[duplicates,0], weights=POIsv1[duplicates,1])
        avgT0 = np.average(POIsv1[duplicates,1], weights=POIsv1[duplicates,1])
        avgD = np.average(POIsv1[duplicates,2], weights=POIsv1[duplicates,1])
        avgZ = np.average(POIsv1[duplicates,3], weights=POIsv1[duplicates,1])
        avgrpRs = np.average(POIsv1[duplicates,4], weights=POIsv1[duplicates,1])
        avgchi2min = np.average(POIsv1[duplicates,5], weights=POIsv1[duplicates,1])
        avgchi2redmin = np.average(POIsv1[duplicates,6], weights=POIsv1[duplicates,1])
        avgSDEraw = np.average(POIsv1[duplicates,7], weights=POIsv1[duplicates,1])
        avgSDE = np.average(POIsv1[duplicates,8], weights=POIsv1[duplicates,1])
        avgsnr = np.average(POIsv1[duplicates,9], weights=POIsv1[duplicates,1])
        avgOED = np.average(POIsv1[duplicates,10], weights=POIsv1[duplicates,1])
        Nocc = np.sum(duplicates)
        POIsv2 = np.vstack([POIsv2, [avgP,avgT0,avgD,avgZ,avgrpRs,avgchi2min,avgchi2redmin,avgSDEraw,avgSDE,avgsnr,avgOED,Nocc]])

    # remove duplicates
    _,inds = np.unique(POIsv2[:,0], return_index=True)
    POIsv3 = POIsv2[inds]
    
    # leave multiples alone because if they surive the iterative TLS,
    # they may be real (e.g. L 98-59)
    ts.vetting.POIs,ts.vetting.T0OIs,ts.vetting.DOIs,ts.vetting.ZOIs,ts.vetting.rpRsOIs,ts.vetting.chi2minOIs,ts.vetting.chi2redminOIs,ts.vetting.SDErawOIs,ts.vetting.SDEOIs,ts.vetting.snrOIs,ts.vetting.oddevendiff_sigma,ts.vetting.NoccurrencesOIs = POIsv3.T
    ts.vetting.vetting_mask = np.ones(POIsv3.shape[0]).astype(bool)



def vet_multiple_sectors(ts):
    '''
    If multiple sectors have been observed, vet planets that were not recovered
    in multiple sectors. 
    '''
    if ts.lc.Nsect <= 1:
        pass
    else:
        ts.vetting.vetting_mask *= ts.vetting.NoccurrencesOIs > 1


        
def vet_odd_even_difference(ts):
    '''
    Check for a significant odd-even difference in the transit depths.
    '''
    ts.vetting.vetting_mask *= ts.vetting.oddevendiff_sigma < 3



def save_planet_parameters(ts):
    outp = np.vstack([ts.vetting.POIs, ts.vetting.T0OIs, ts.vetting.DOIs,
                      ts.vetting.ZOIs, ts.vetting.rpRsOIs, ts.vetting.chi2minOIs,
                      ts.vetting.chi2redminOIs, ts.vetting.SDErawOIs,
                      ts.vetting.SDEOIs, ts.vetting.snrOIs,
                      ts.vetting.oddevendiff_sigma, ts.vetting.NoccurrencesOIs,
                      ts.vetting.vetting_mask]).T
    df = pd.DataFrame(outp, columns=['P','T0','duration_hrs','depth_ppt','rpRs',
                                     'chi2min','chi2redmin','SDEraw','SDE','snr',
                                     'oddevendiff_sig','Noccurrences','vetted?'])
    df = df.sort_values('SDE', ascending=False)
    df.to_csv('%s/MAST/TESS/TIC%i/planetparams.csv'%(cs.repo_dir, ts.tic), index=False)
