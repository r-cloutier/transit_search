from tls_object import *
import numpy as np
import glob, pandas


fs=np.sort(glob.glob('MAST/TESS/TIC*/TESSLC_planetsearch'))

outarr = np.zeros((fs.size,6))

for i,f in enumerate(fs):
    print(i/fs.size)
    ts=loadpickle(f)
    try:
        outarr[i] = int(ts.tic), ts.star.Tmag, ts.star.Teff, ts.star.Ms, ts.star.Rs, ts.star.Prot
    except AttributeError:
        pass

df = pandas.DataFrame(outarr, columns=['TIC','Tmag','Teff','Ms','Rs','Prot'])
df.to_csv('/n/home10/rcloutier/TLS/MAST/TESS/Prots.csv')
