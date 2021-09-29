from imports import *
from scipy.signal import medfilt


def detrend_LC_1sector(bjd, fnorm, window_length_days=0.5):
    # get window length for running median and ensure that it's an odd number
    N = np.round(window_length_days / np.median(np.diff(bjd)))
    N = int(N+1) if N % 2  == 0 else int(N)
    model = medfilt(fnorm, N)
    fdetrend = fnorm / model
    return fdetrend, model


def detrend_all_sectors(bjd, fnorm, sectors, window_length_days=0.5,
                        sig=5):
    assert bjd.size == fnorm.size
    assert bjd.size == sectors.size

    fdetrend, model, mask = np.zeros_like(bjd), np.zeros_like(bjd), np.zeros_like(bjd)
    for s in np.unique(sectors):
        # detrend this sector's LC
        g = sectors == s
        fdetrend[g], model[g]  = detrend_LC_1sector(bjd[g], fnorm[g], window_length_days)
        
        # mask out outliers via sigma clip
        mask[g] = sigma_clip(bjd[g], fdetrend[g], sig=sig)
        
    return fdetrend, model, mask.astype(bool)


MAD = lambda arr: np.nanmedian(abs(arr - np.nanmedian(arr)))


def sigma_clip(bjd, f, sig=5):
    mask = abs(f - np.nanmedian(f)) < sig*MAD(f)
    return mask

