import os, pickle
import constants as cs


def loadpickle(fname):
    fObj = open(fname, 'rb')
    self = pickle.load(fObj)
    fObj.close()
    return self


class transit_search:

    def __init__(self, tic, fname):
        self.tic, self.DONE = int(tic), False
        dir_full = '%s/MAST/TESS/TIC%i'%(cs.repo_dir, tic)
        assert os.path.exists(dir_full)
        self.fname_full = '%s/%s'%(dir_full, fname)

        # initialize inner classes
        self.lc = self.LC()
        self.star = self.STAR()
        self.gls = self.GLS()
        self.tls = self.TLS()
        self.vetting = self.VETTING()
        self.injrec = self.INJREC()

    def pickleobject(self):
        fObj = open(self.fname_full, 'wb')
        pickle.dump(self, fObj)
        fObj.close()

        
    class LC:
        pass
    class STAR:
        pass
    class GLS:
        pass
    class TLS:
        pass
    class VETTING:
        pass
    class INJREC:
        pass
