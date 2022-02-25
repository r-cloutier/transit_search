import numpy as np
from scipy.signal import medfilt, savgol_filter


def detrend_LC_1sector(bjd, fnorm, window_length_hrs=12):
    # get window length for running median and ensure that it's an odd number
    N = np.round(window_length_hrs / (np.median(np.diff(bjd)[:100])*24))
    N = int(N+1) if N % 2  == 0 else int(N)
    #detrend_model = medfilt(fnorm, N)
    detrend_model = savgol_filter(fnorm, N, 1)
    return fnorm/detrend_model, detrend_model


def detrend_all_sectors(bjd, fnorm, sectors, window_length_hrs=12, sig=5):
    assert bjd.size == fnorm.size
    assert bjd.size == sectors.size

    fdetrend, detrend_model, mask = np.zeros_like(bjd), np.zeros_like(bjd), np.zeros_like(bjd)
    for s in np.unique(sectors):
        # detrend this sector's LC
        g = sectors == s
        fdetrend[g], detrend_model[g] = detrend_LC_1sector(bjd[g], fnorm[g], window_length_hrs)
        
        # mask out outliers via sigma clip
        mask[g] = sigma_clip(bjd[g], fdetrend[g], sig=sig)
        
    return fdetrend, detrend_model, mask.astype(bool)


MAD = lambda arr: np.nanmedian(abs(arr - np.nanmedian(arr)))


def sigma_clip(bjd, f, sig=5):
    mask = abs(f - np.nanmedian(f)) < sig*MAD(f)
    return mask


