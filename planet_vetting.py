import numpy as np


def get_POIs(ts):
    '''
    Given a set of tls result dictionaries (i.e. from multiple TLS runs and/or
    multiple sectors), consolidate periods of interest. 
    '''
    # collect all periods
    POIsv1 = np.zeros((0,2))   # (period, SDE)
    for res in ts.tls.__dict__.keys():
        POIsv1 = np.vstack([POIsv1, [getattr(ts.tls,res).period,
                                     getattr(ts.tls,res).SDE_raw]])
    POIsv1 = POIsv1[np.argsort(POIsv1,0)[:,0]]
    
    # identify duplicates
    POIsv2 = np.zeros((0,3))  # (avg(P), avg(SDE), Noccurrences)
    for p in POIsv1[:,0]:
        duplicates = np.isclose(POIsv1[:,0], p, rtol=cs.P_duplicate_fraction)
        avgP = np.average(POIsv1[duplicates,0], weights=POIsv1[duplicates,1])
        avgSDE = np.average(POIsv1[duplicates,1], weights=POIsv1[duplicates,1])
        Nocc = np.sum(duplicates)
        POIsv2 = np.vstack([POIsv2, [avgP,avgSDE,Nocc]])

    # remove duplicates
    _,inds = np.unique(POIsv2[:,0], return_index=True)
    POIsv3 = POIsv2[inds]
    
    # leave multiples alone because if they surive the iterative TLS,
    # they may be real (e.g. L 98-59)
    ts.vettting.POIs,ts.vettting.SDEOIs,ts.vettting.NoccurrencesOIs = POIsv3.T
    ts.vetting.vetting_mask = np.ones(POIsv3.shape[0]).astype(bool)



def vet_multiple_sectors(ts):
    '''
    If multiple sectors have been observed, vet planets that were not recovered
    in multiple sectors. 
    '''
    Nsect = len(ts.tls.__dict__) / cs.Nplanets_max
    if Nsect <= 1:
        pass
    else:
        ts.vetting.vetting_mask *= ts.vetting.NoccurrencesOIs > 1
        
