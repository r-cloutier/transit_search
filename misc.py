from operator import itemgetter
from itertools import groupby

def get_consecutive_sectors(sectors):
    '''
    Returns a list of lists where each element is a list of consecutive sectors.
    
    E.g. for a target observed in sectors 1,2,3,13,14,20, this function will return

    [[1, 2, 3], [13, 14], [20]]
    '''
    ranges=[]
    for k,g in groupby(enumerate(sectors),lambda x:x[0]-x[1]):
        group = (map(itemgetter(1),g))
        group = list(map(int,group))
        ranges.append(group if len(group) == 1 else list(range(group[0],group[-1]+1)))
    return ranges
