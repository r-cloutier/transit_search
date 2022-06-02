global t0, plot_dir, minsector, maxsector, P_duplicate_fraction, Nplanets_min, SDEthreshold, Pgrid, Rpgrid, bgrid
  
# time offset for plotting
t0 = 2457e3

# base directory that holds everything
repo_dir = '/n/home10/rcloutier/TLS'

# fractional threshold for identifying duplicate periods in the TLS
P_duplicate_fraction = 0.02

# lowest TESS sector to search for data
minsector = 1

# highest TESS sector to search for data
maxsector = 55

# minimum number of planets to search for with the TLS
Nplanets_max = 3

# stop the TLS search when the max SDE value is less than this value
SDEthreshold = 5

# minimum power in Gls to assign to stellar rotation
minGlspwr = .1

# define grid for injection-recovery
Pgrid = .1, 30   # days
Rpgrid = .1, 4   # Rearth
bgrid = 0, .9
ampgrid = .1, 100  # ppt
Protgrid = .1, 100  # days
