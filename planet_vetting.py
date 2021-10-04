import numpy as np


def vet_planets(ts):
    '''
    Main function to identify potential planetary signals and to vet them
    using the criteria defined by the functions herein.
    '''
    # remove duplicate periodicities + save planet and vetting parameters  
    get_POIs(ts)

    # apply vetting criteria
    vet_multiple_sectors(ts)
    vet_odd_even_difference(ts)
    
    

def get_POIs(ts):
    '''
    Given a set of tls result dictionaries (i.e. from multiple TLS runs and/or
    multiple sectors), consolidate periods of interest. 
    '''
    # collect all periods
    # (period, T0, duration, depth, oddevensigma, SDE)
    POIsv1 = np.zeros((0,6))
    for res in ts.tls.__dict__.keys():
        POIsv1 = np.vstack([POIsv1, [getattr(ts.tls,res).period,
                                     getattr(ts.tls,res).T0,
                                     getattr(ts.tls,res).duration*24,
                                     1-getattr(ts.tls,res).depth,
                                     getattr(ts.tls,res).odd_even_mismatch,
                                     getattr(ts.tls,res).SDE]])
    POIsv1 = POIsv1[np.argsort(POIsv1,0)[:,0]]
    
    # identify duplicates
    # (avg(P),avg(T0),avg(duration),avg(depth),avg(odd/even),avg(SDE),
    # Noccurrences)
    POIsv2 = np.zeros((0,7))
    for p in POIsv1[:,0]:
        duplicates = np.isclose(POIsv1[:,0], p, rtol=cs.P_duplicate_fraction)
        avgP = np.average(POIsv1[duplicates,0], weights=POIsv1[duplicates,1])
        avgT0 = np.average(POIsv1[duplicates,1], weights=POIsv1[duplicates,1])
        avgD = np.average(POIsv1[duplicates,2], weights=POIsv1[duplicates,1])
        avgZ = np.average(POIsv1[duplicates,3], weights=POIsv1[duplicates,1])
        avgoddeven = np.average(POIsv1[duplicates,4], weights=POIsv1[duplicates,1])
        avgSDE = np.average(POIsv1[duplicates,5], weights=POIsv1[duplicates,1])
        Nocc = np.sum(duplicates)
        POIsv2 = np.vstack([POIsv2, [avgP,avgT0,avgD,avgZ,avgoddeven,avgSDE,Nocc]])

    # remove duplicates
    _,inds = np.unique(POIsv2[:,0], return_index=True)
    POIsv3 = POIsv2[inds]
    
    # leave multiples alone because if they surive the iterative TLS,
    # they may be real (e.g. L 98-59)
    ts.vetting.POIs,ts.vetting.T0OIs,ts.vetting.DOIs,ts.vetting.ZOIs,ts.vetting.oddevensigmaOIs,ts.vetting.SDEOIs,ts.vetting.NoccurrencesOIs = POIsv3.T
    ts.vetting.vetting_mask = np.ones(POIsv3.shape[0]).astype(bool)




def vet_multiple_sectors(ts):
    '''
    If multiple sectors have been observed, vet planets that were not recovered
    in multiple sectors. 
    '''
    ##Nsect = int(np.sum(['_1_' in k for k in ts.tls.__dict__.keys()]))
    if ts.lc.Nsect <= 1:
        pass
    else:
        ts.vetting.vetting_mask *= ts.vetting.NoccurrencesOIs > 1



def vet_odd_even_difference(ts):
    '''
    Check for the significance of the depth difference between odd and even
    transits.
    '''
    for sigma in ts.vetting.oddevensigmaOIs:
        ts.vetting.vetting_mask *= ts.vetting.oddevensigmaOIs < 3
