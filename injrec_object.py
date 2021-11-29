import os, pickle
import constants as cs


def loadpickle(fname):
    fObj = open(fname, 'rb')
    self = pickle.load(fObj)
    fObj.close()
    return self


class injection_recovery:

    def __init__(self, Tmagmin, Tmagmax, fname):
        self.Tmagmin, self.Tmagmax = float(Tmagmin), float(Tmagmax)
        self.DONE = False
        dir_full = '%s/MAST/TESS'%cs.repo_dir
        assert os.path.exists(dir_full)
        self.fname_full = '%s/%s'%(dir_full, fname)

    def pickleobject(self):
        fObj = open(self.fname_full, 'wb')
        pickle.dump(self, fObj)
        fObj.close()
