from tls_object import *
import numpy as np
import glob, pandas

fs=np.sort(glob.glob('MAST/TESS/TIC*/TESSLC_planetsearch'))

outarr = np.zeros((fs.size,7))

for i,f in enumerate(fs):
    print(i/fs.size)
    ts=loadpickle(f)
    try:
        outarr[i] = int(ts.tic), ts.star.Tmag, ts.star.Teff, ts.star.Ms, ts.star.Rs, ts.star.Prot_gls, ts.star.Prot_tls
    except AttributeError:
        pass

df = pandas.DataFrame(outarr, columns=['TIC','Tmag','Teff','Ms','Rs','Prot_gls','Prot_tls'])
df.to_csv('/n/home10/rcloutier/TLS/Protspreadsheet.csv')
