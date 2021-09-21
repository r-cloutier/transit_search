import os, requests, glob, pdb
import numpy as np
from bs4 import BeautifulSoup
from astropy.io import fits
from astroquery.mast import Catalogs


def listFD(url, ext=''):
    page = requests.get(url).text
    soup = BeautifulSoup(page, 'html.parser')
    return np.array([url + '/' + node.get('href') for node in soup.find_all('a') if node.get('href').endswith(ext)])


def read_TESS_data(tic, minsector=1, maxsector=55, quality_cut=True,
                   use_20sec=True):
    # make directories
    localdir = os.getcwd()
    try:
        os.mkdir('%s/MAST'%localdir)
    except OSError:
        pass
    try:
        os.mkdir('%s/MAST/TESS'%localdir)
    except OSError:
        pass
    folder2 = '%s/MAST/TESS/TIC%i'%(localdir,tic)
    try:
         os.mkdir(folder2)
    except OSError:
         pass

    # setup directories to fetch the data
    tic_str = '%.16d'%int(tic)
    tid1 = tic_str[:4]
    tid2 = tic_str[4:8]
    tid3 = tic_str[8:12]
    tid4 = tic_str[12:]

    url = 'https://archive.stsci.edu/missions/tess/tid/'

    # download lc fits file if not already
    fnames = np.array(glob.glob('%s/tess*.fits'%folder2))
    for j in range(int(minsector), int(maxsector)+1):
        sctr = 's%.4d'%j

        # check that we don't already have the LC file from this sector
        if not np.any([sctr in f for f in fnames]):
            folder = '%s/%s/%s/%s/%s/'%(sctr,tid1,tid2,tid3,tid4)
            fs = listFD(url+folder, ext='lc.fits')
            for i in range(fs.size):
                os.system('wget %s'%fs[i])
                fname = str(fs[i]).split('/')[-1]
                if os.path.exists(fname):
                    os.system('mv %s %s'%(fname, folder2))
   
    # get data from fits files and remove duplicate sectors
    fnames = np.sort(np.array(glob.glob('%s/tess*lc.fits'%folder2)))
    fnamesv2 = remove_duplicate_sectors(fnames, folder2,
                                        keep_2min=not use_20sec)

    bjd, fnorm, efnorm, sectors, qual_flags, texps = np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0)
    for i in range(fnamesv2.size):
        hdus = fits.open(fnamesv2[i])
        bjd = np.append(bjd, hdus[1].data['TIME'] + 2457000)
        ftmp = hdus[1].data['PDCSAP_FLUX']
        eftmp = hdus[1].data['PDCSAP_FLUX_ERR']
        eftmp /= np.nanmedian(ftmp)
        ftmp /= np.nanmedian(ftmp)
        fnorm = np.append(fnorm, ftmp)
        efnorm = np.append(efnorm, eftmp)
        sectors = np.append(sectors, np.repeat(hdus[0].header['SECTOR'], ftmp.size))
        qual_flags = np.append(qual_flags, hdus[1].data['QUALITY'])

        hdr = hdus[1].header
        texps = np.append(texps, np.repeat(hdr["FRAMETIM"]*hdr["NUM_FRM"] / (60*60*24), ftmp.size))
        
    bjd = np.ascontiguousarray(bjd)
    fnorm = np.ascontiguousarray(fnorm)
    efnorm = np.ascontiguousarray(efnorm)
    sectors = np.ascontiguousarray(sectors)
    qual_flags = np.ascontiguousarray(qual_flags)
    texps = np.ascontiguousarray(texps)

    # restrict quality flags
    if quality_cut:
        g = (qual_flags == 0) & (np.isfinite(fnorm)) & (np.isfinite(efnorm))
        bjd, fnorm, efnorm, sectors, qual_flags, texps = bjd[g], fnorm[g], efnorm[g], sectors[g], qual_flags[g], texps[g]

    return bjd, fnorm, efnorm, sectors, qual_flags, texps




def remove_duplicate_sectors(fnames, folder2, keep_2min=False):
    # given a list of lc.fits files, search for duplicate sectors and only keep the 20-second data or 2-minute data (if desired)
    fnamesv2 = np.copy(fnames)
    sectors,counts = np.unique([int(f.split('-s')[1].split('-')[0]) for f in fnamesv2], return_counts=True)
    if np.any(counts > 1):
        for s in sectors[counts>1]:
            glob_str = '%s/tess*-s%.4d*-lc.fits'%(folder2,s) if keep_2min else '%s/tess*-s%.4d*_lc.fits'%(folder2,s)
            fnamesv2 = np.delete(fnamesv2, np.where(np.in1d(fnamesv2, np.array(glob.glob(glob_str))))[0])
    return fnamesv2



def get_star(tic):
    star_info = Catalogs.query_object("TIC %i"%tic, catalog="TIC")
    star_info = star_info[star_info['ID'] == tic]

    star_dict = {'tic': tic, 'ra': star_info['ra'],
                 'dec': star_info['dec'], 'GBPmag': star_info['gaiabp'],
                 'e_GBPmag': star_info['e_gaiabp'], 'GRPmag': star_info['gaiarp'],
                 'e_GRPmag': star_info['e_gaiarp'], 'TESSmag': star_info['Tmag'],
                 'Jmag': star_info['Jmag'], 'e_Jmag': star_info['e_Jmag'],
                 'Hmag': star_info['Hmag'], 'e_Hmag': star_info['e_Hmag'],
                 'Kmag': star_info['Kmag'], 'e_Kmag': star_info['e_Kmag'],
                 'par': star_info['plx'], 'e_par': star_info['e_plx'],
                 'dist': star_info['d'], 'ehi_dist': star_info['epos_dist'],
                 'elo_dist': star_info['eneg_dist'], 'mu': np.nan,
                 'ehi_mu': np.nan, 'elo_mu': np.nan,
                 'AK': np.nan, 'e_AK': np.nan,
                 'MK': np.nan, 'ehi_MK': np.nan,
                 'elo_MK': np.nan, 'Rs': star_info['rad'],
                 'ehi_Rs': star_info['e_rad'], 'elo_Rs': star_info['e_rad'],
                 'Teff': star_info['Teff'], 'ehi_Teff': star_info['e_Teff'],
                 'elo_Teff': star_info['e_Teff'], 'Ms': star_info['mass'],
                 'ehi_Ms': star_info['e_mass'], 'elo_Ms': star_info['e_mass'],
                 'logg': star_info['logg'], 'ehi_logg': star_info['e_logg'],
                 'elo_logg': star_info['e_logg']}

    return star_dict


def convert2exoplanet(bjd, fnorm, efnorm):
    ref_time = np.nanmean([np.nanmax(bjd), np.nanmin(bjd)])
    return bjd-ref_time, 1e3*(fnorm-1), 1e3*efnorm, ref_time


def convert2norm(bjd_shift, fexo, efexo, ref_time):
    return bjd_shift+ref_time, 1e-3*fexo+1, 1e-3*efexo

