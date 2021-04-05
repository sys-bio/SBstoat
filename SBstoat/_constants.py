"""Constants used within SBstoat."""

#  Minimizer methods
METHOD_DIFFERENTIAL_EVOLUTION = "differential_evolution"
METHOD_BOTH = "both"
METHOD_LEASTSQ = "leastsq"
METHOD_FITTER_DEFAULTS = [METHOD_DIFFERENTIAL_EVOLUTION, METHOD_LEASTSQ]
METHOD_BOOTSTRAP_DEFAULTS = [METHOD_LEASTSQ]
# Keywords
MAX_NFEV = "max_nfev"
MAX_NFEV_DFT = 100
PARAMS = "params"
# Kwargs
LOGGER = "logger"
# Column names
COUNT = "count"
FOLD = "fold"
MEAN = "mean"
PREDICTED = "predicted"
PARAMETER = "parameter"
RSQ = "rsq"
SCORE = "score"
STD = "std"
TIME = "time"
TRUE = "true"
VALUE = "value"
# Parameter class
PARAMETER_LOWER_BOUND = 0
PARAMETER_UPPER_BOUND = 10
