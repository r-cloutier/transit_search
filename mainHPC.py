import sys, numpy
import planet_search as ps

input_fname = sys.argv[1]

tics = numpy.loadtxt(input_fname)[1:]

for i,tic in enumerate(tics):
    ts = ps.run_full_planet_search(tic, use_20sec=False)
