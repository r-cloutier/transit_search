global t0, plot_dir, minsector, maxsector, P_duplicate_fraction, Nplanets_max
  
# time offset for plotting
t0 = 2457e3

# base directory that holds everything
repo_dir = '/Users/ryancloutier/Research/TLS'

# fractional threshold for identifying duplicate periods in the TLS
P_duplicate_fraction = 0.02

# lowest TESS sector to search for data
minsector = 1

# highest TESS sector to search for data
maxsector = 41

# maximum number of planets to search for with the TLS
Nplanets_max = 3
