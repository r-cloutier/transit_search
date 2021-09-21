from imports import *
from scipy.signal import medfilt


def detrend_LC_1sector(bjd, fnorm, window_length_days=0.5):
    # get window length for running median and ensure that it's an odd number
    N = np.round(np.nanmax(bjd) - np.nanmin(bjd)) / window_length_days
    N = int(N+1) if N % 2  == 0 else int(N)
    fdetrend = fnorm / medfilt(fnorm, N)
    return fdetrend


def detrend_all_sectors(bjd, fnorm, sectors, window_length_days=0.5,
                        sig=5):
    assert bjd.size == fnorm.size
    assert bjd.size == sectors.size

    fdetrend, mask = np.zeros_like(bjd), np.zeros_like(bjd)
    for s in np.unique(sectors):
        # detrend this sector's LC
        g = sectors == s
        fdetrend[g] = detrend_LC_1sector(bjd[g], fnorm[g], window_length_days)
        
        # mask out outliers via sigma clip
        mask[g] = sigma_clip(bjd[g], fdetrend[g], sig=sig)
        
    return fdetrend, mask.astype(bool)


MAD = lambda arr: np.nanmedian(abs(arr - np.nanmedian(arr)))


def sigma_clip(bjd, f, sig=5):
    mask = abs(f - np.nanmedian(f)) < sig*MAD(f)
    return mask

