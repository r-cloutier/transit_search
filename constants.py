global t0, plot_dir, minsector, maxsector, P_duplicate_fraction, Ntransits_min, Nplanets_max, SNRthreshold, SDEthreshold, Pgrid, Rpgrid, bgrid, DONEcheck_version
  
# time offset for plotting
t0 = 2457e3

# base directory that holds everything
repo_dir = '/n/holylabs/LABS/charbonneau_lab/Users/rcloutier/TLS'

# fractional threshold for identifying duplicate periods in the TLS
P_duplicate_fraction = 0.02

# lowest TESS sector to search for data
minsector = 1

# highest TESS sector to search for data
maxsector = 55

# minimum number of transits
Ntransits_min = 3

# maximum number of planets to search for with the TLS
Nplanets_max = 3

# stop the TLS search when the max SDE value is less than this value
SNRthreshold = 2
SDEthreshold = 5

# minimum power in Gls to assign to stellar rotation
minGlspwr = .1

# define grid for injection-recovery
Pgrid = .1, 30   # days
Rpgrid = .1, 4   # Rearth
bgrid = 0, .9
ampgrid = .1, 100  # ppt
Protgrid = .1, 100  # days

# a unique numerical identifier set in planet_search to check if the most up-to-date version has been run
DONEcheck_version = 1   # 2022-06-27
