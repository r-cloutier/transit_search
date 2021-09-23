import os, pickle
import constants as cs


def loadpickle(fname):
    fObj = open(fname, 'rb')
    self = pickle.load(fObj)
    fObj.close()
    return self


class transit_search:

    def __init__(self, tic, fname):
        self.tic = int(tic)
        dir_full = '%s/MAST/TESS/TIC%i'%(cs.repo_dir, tic)
        assert os.path.exists(dir_full)
        self.fname_full = '%s/%s'%(dir_full, fname)

        # initialize inner classes
        self.lc = self.LC()
        self.star = self.STAR()
        self.tls = self.TLS()
        self.vetting = self.VETTING()


    def pickleobject(self):
        fObj = open(self.fname_full, 'wb')
        pickle.dump(self, fObj)
        fObj.close()

        
    class LC:
        pass
    class STAR:
        pass
    class TLS:
        pass
    class VETTING:
        pass
                
