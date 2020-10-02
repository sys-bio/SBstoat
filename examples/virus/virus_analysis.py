print("# In[1]:")


# Imports
import lmfit
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import SBstoat
import tellurium as te
import matplotlib
import pickle
#get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
matplotlib.use('TkAgg')


print("# In[24]:")


IS_PLOT = True
VIRUS = "V"
DIR = "/home/ubuntu/SBstoat/examples/virus"
PICKLE_FILE = os.path.join(DIR, "virus.pcl")
INPUT_FILE =  os.path.join(DIR, "Influenza.csv")
NUM_BOOTSTRAP_ITERATIONS = 10000
PARAM_BETA = "beta"
PARAM_KAPPA = "kappa"
PARAM_P = "p"
PARAM_C = "c"
PARAM_DELTA = "delta"
PARAMS = [PARAM_BETA, PARAM_KAPPA, PARAM_P, PARAM_C, PARAM_DELTA]
TIME = "time"


# ## Antimony Model

print("# In[3]:")


ANTIMONY_MODEL  = '''
    // Equations
    E1: T -> E ; beta*T*V ; // Target cells to exposed
    E2: E -> I ; kappa*E ;  // Exposed cells to infected
    E3: -> V ; p*I ;        // Virus production by infected cells
    E4: V -> ; c*V ;        // Virus clearance
    E5: I -> ; delta*I      // Death of infected cells    

    // Parameters - from the Influenza article,
        
    beta = 3.2e-5;  // rate of transition of target(T) to exposed(E) cells, in units of 1/[V] * 1/day
    kappa = 4.0;    // rate of transition from exposed(E) to infected(I) cells, in units of 1/day
    delta = 5.2;    // rate of death of infected cells(I), in units of 1/day
    p = 4.6e-2;     // rate virus(V) producion by infected cells(I), in units of [V]/day
    c = 5.2;        // rate of virus clearance, in units of 1/day

    // Initial conditions
    T = 4E+8 // estimate of the total number of susceptible epithelial cells
             // in upper respiratory tract)
    E = 0
    I = 0
    V = 0.75 // the dose of virus in TCID50 in Influenza experiment; could be V=0 and I = 20 instead for a natural infection

'''


# ## Functions

print("# In[4]:")


def extractPatientData(timeseries, patient):
     newTimeseries = timeseries.copy()
     newTimeseries[VIRUS] = newTimeseries[patient]
     return newTimeseries.subsetColumns([VIRUS])


# ## Data Setup

print("# In[5]:")


"""
Transform the input data
"""

# Convert input file to correct format
dataDF = pd.read_csv(INPUT_FILE, header=None)
dataDF = dataDF.transpose()
dataDF.index.name = "time"
patientDct = {p: "P%d" % (p+1) for p in range(6)}
dataDF = dataDF.rename(columns=patientDct)
observedTS = SBstoat.NamedTimeseries(dataframe=dataDF)
print(observedTS)


print("# In[35]:")


# Compare baseline simulation with observed values

rr = te.loada(ANTIMONY_MODEL)
estimates = rr.simulate(0, 6)
v_estimate = np.log10(estimates["[V]"])
fig, ax = plt.subplots(1, 1)
ax.plot(estimates["time"], v_estimate)
COLORS = ["r", "grey", "b", "g", "yellow", "pink"]
for idx, patient in enumerate(observedTS.colnames):
    color = COLORS[idx]
    ax.scatter(observedTS[TIME], observedTS[patient], color=color)
legends = ["fitted"]
legends.extend(observedTS.colnames)
plt.legend(legends)


# ## Construct Parameter Fittings
# This is a computationally intensive procedure. So, results from a past parameter fit are saved in PICKLE_FILE.
# The plots constructed in the next section

print("# In[7]:")


# Do some fits
# Specify the parameter values
parameterDct = {}
SBstoat.ModelFitter.addParameter(parameterDct, "beta", 0, 10e-5, 3.2e-5)
SBstoat.ModelFitter.addParameter(parameterDct, "kappa", 0, 10, 4.0)
SBstoat.ModelFitter.addParameter(parameterDct, "delta", 0, 10, 5.2)
SBstoat.ModelFitter.addParameter(parameterDct, "p", 0, 1, 4.6e-2)
SBstoat.ModelFitter.addParameter(parameterDct, "c", 0, 10, 5.2)


print("# In[8]:")


def transformData(timeseries):
    """
    Changes the timeseries to log units
    """
    arr = np.array([1 if v < 1 else v
          for v in timeseries[VIRUS]])
    timeseries[VIRUS] = np.log10(arr)
    return timeseries


print("# In[9]:")


def transformDataArr(timeseries):
    """
    Changes the timeseries to log units
    """
    arr = np.array([1 if v < 1 else v
          for v in timeseries[VIRUS]])
    return np.log10(arr)


print("# In[37]:")


# Create bootstrapped fitters for all patients
def mkFitter(patient, isBootstrap=True):
    """
    Creates a fitter for the patient and runs
    bootsrapping.

    Parameters
    ----------
    patient: str
    
    Returns
    -------
    ModelFitter
    """
    parameterDct = {}
    SBstoat.ModelFitter.addParameter(parameterDct, "beta", 0, 10e-5, 3.2e-5)
    SBstoat.ModelFitter.addParameter(parameterDct, "kappa", 0, 10, 4.0)
    SBstoat.ModelFitter.addParameter(parameterDct, "delta", 0, 10, 5.2)
    SBstoat.ModelFitter.addParameter(parameterDct, "p", 0, 1, 4.6e-2)
    SBstoat.ModelFitter.addParameter(parameterDct, "c", 0, 10, 5.2)
    # Obtain the input data
    patientObservedTS = extractPatientData(observedTS, patient)
    fittedDataTransformDct = {VIRUS: transformDataArr}  # do fit in log units

    # Fit parameters to ts1
    fitter = SBstoat.ModelFitter(ANTIMONY_MODEL,
        patientObservedTS, ["beta","kappa","delta","p","c"],
        fittedDataTransformDct=fittedDataTransformDct,
        parameterDct=parameterDct)
    fitter.fitModel()
    # Do the bootstrap
    reportInterval = int(NUM_BOOTSTRAP_ITERATIONS/10)
    if isBootstrap:
        fitter.bootstrap(numIteration=NUM_BOOTSTRAP_ITERATIONS,
              reportInterval=reportInterval)
    #
    return fitter


# ## Do a Single Fit

print("# In[41]:")


fitter = mkFitter("P1", isBootstrap=False)
print(fitter.params)


print("# In[42]:")


fitter.plotFitAll(numRow=2, numCol=2, ylabel="log10(V)")
