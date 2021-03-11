"""Constants used within SBstoat."""

#  Minimizer methods
METHOD_DIFFERENTIAL_EVOLUTION = "differential_evolution"
METHOD_BOTH = "both"
METHOD_LEASTSQ = "leastsq"
METHOD_FITTER_DEFAULTS = [METHOD_DIFFERENTIAL_EVOLUTION, METHOD_LEASTSQ]
METHOD_BOOTSTRAP_DEFAULTS = [METHOD_LEASTSQ,
      METHOD_DIFFERENTIAL_EVOLUTION, METHOD_LEASTSQ]
# Keywords
MAX_NFEV = "max_nfev"
MAX_NFEV_DFT = 100
# Kwargs
LOGGER = "logger"
# Dataframe columns
FOLD = "fold"
MEAN = "mean"
PREDICTED = "predicted"
PARAMETER = "parameter"
RSQ = "rsq"
SCORE = "score"
STD = "std"
TRUE = "true"
VALUE = "value"
