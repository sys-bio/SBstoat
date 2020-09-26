#!/usr/bin/env python
# coding: utf-8


"""
Analysis of the virus model.

 We will fit each of these patients separately, obtaining different values of parameters.

 ## Fitting

 There are 6 patients with one value to fit. So, there are 6 different fits.


 Influenza, SARS and SARS-CoV3 in Tellurium

## A simple target cell-limited model, T ==> E ==> I --> produce V:
 T - number of target cells
 E - number of exposed cells (virus replicating inside, not yet spreading virus)
 I - number of infected cells (active virus production)
 V - viral titre, in units of TCID50/ml of biofluid wash (for Influenza)

# The ODEs
 dT/dt = - beta*T*V
 dE/dt =   beta*T*V - kappa*E;
 dI/dt =   kappa * E - delta*I;
 dV/dt =   p*y(I) - c*y(V);

 All viral data is in log10(load),...
 log10(load predicted by model) may be needed for data fitting


Influenza.csv
 Influenza A data - 5 patients
 viral levels in log10(TCID50 / ml of nasal wash)
 time in days since volunteer exposure
 each line in the array is an individual volunteer



SARS_CoV2_sputum.csv and SARS_CoV2_nasal.csv
 SARS-CoV-2 data - 9 patients,
 for each patient - viral loads from lungs (sputum) and from nasal cavity (swab)
 viral levels in log10(RNA copies / ml sputum), ...
 respectively log10(RNA copies / nasal swab)
 time in days since symptoms onset
 corresponding lines in the two arrays belong to an individual patient



SARS.csv
 SARS data recorded from 12 patients;
 included them just for comparison, probably too few datapoints for model inference
 viral levels in log10(RNA copies / ml of nasopharingeal aspirate)
 time - only three samples per patient, at 5, 10 and 15 days post symptoms onset
"""

# Imports
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import SBstoat
import tellurium as te
import matplotlib
import pickle
matplotlib.use('TkAgg')

IS_PLOT = True
VIRUS = "V"
PICKLE_FILE = "virus.pcl"
NUM_BOOTSTRAP_ITERATIONS = 10000
PARAM_BETA = "beta"
PARAM_KAPPA = "kappa"
PARAM_P = "p"
PARAM_C = "c"
PARAM_DELTA = "delta"
PARAMS = [PARAM_BETA, PARAM_KAPPA, PARAM_P, PARAM_C, PARAM_DELTA]

"""
Transform the input data
"""

input_file = "Influenza.csv"

# Convert input file to correct format
dataDF = pd.read_csv(input_file, header=None)
dataDF = dataDF.transpose()
dataDF.index.name = "time"
patientDct = {p: "P%d" % (p+1) for p in range(6)}
dataDF = dataDF.rename(columns=patientDct)
observedTS = SBstoat.NamedTimeseries(dataframe=dataDF)
print(observedTS)

# Extract data in the correct format for fitting
def extractPatientData(timeseries, patient):
     newTimeseries = timeseries.copy()
     newTimeseries[VIRUS] = 10**newTimeseries[patient]
     return newTimeseries.subsetColumns([VIRUS])

def transformData(timeseries):
    """
    Changes the timeseries to log units
    """
    arr = np.array([1 if v < 1 else v 
          for v in timeseries[VIRUS]])
    timeseries[VIRUS] = np.log10(arr)
    return timeseries


# The model

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


# Run the simulation

rr = te.loada(ANTIMONY_MODEL)
estimates = rr.simulate()
v_estimate = np.log10(estimates["[V]"])
#plt.plot(estimates["time"], v_estimate)


# Do some fits
# Specify the parameter values
parameterDct = {}
SBstoat.ModelFitter.addParameter(parameterDct, "beta", 0, 10e-5, 3.2e-5)
SBstoat.ModelFitter.addParameter(parameterDct, "kappa", 0, 10, 4.0)
SBstoat.ModelFitter.addParameter(parameterDct, "delta", 0, 10, 5.2)
SBstoat.ModelFitter.addParameter(parameterDct, "p", 0, 1, 4.6e-2)
SBstoat.ModelFitter.addParameter(parameterDct, "c", 0, 10, 5.2)


# Plot fits for each patient
patients = [p+1 for p in patientDct.keys()]
fig, axes = plt.subplots(5,1, figsize=(12,10))
fittedDataTransformDct = {VIRUS: transformData}
for pos, param in enumerate(PARAMS):
    singleObservedTS = extractPatientData(observedTS, "P4")
    fitter = SBstoat.ModelFitter(ANTIMONY_MODEL,
        singleObservedTS, ["beta","kappa","delta","p","c"],
        parameterDct=parameterDct)
    fitter.fitModel()
    print("\n\n***Analysis for patient %s***\n" % patientDct[pos])
    print(fitter.reportFit())
    fitter.fitModel()
    if IS_PLOT:
        fitter.observedTS = transformData(fitter.observedTS)
        fitter.fittedTS = transformData(fitter.fittedTS)
        fitter.residualsTS = transformData(fitter.residualsTS)
        fitter.plotFitAll(numRow=2, numCol=2, ylabel="log10(V)")
        fitter.plotResiduals(numRow=2, numCol=2, ylabel="log10(V)")


# Create bootstrapped fitters for all patients
def mkFitter(patient):
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
    fittedDataTransformDct = {VIRUS: transformData}
    
    # Fit parameters to ts1
    fitter = SBstoat.ModelFitter(ANTIMONY_MODEL,
        patientObservedTS, ["beta","kappa","delta","p","c"],
        parameterDct=parameterDct)
    fitter.fitModel()
    # Do the bootstrap
    reportInterval = int(NUM_BOOTSTRAP_ITERATIONS/10)
    fitter.bootstrap(numIteration=NUM_BOOTSTRAP_ITERATIONS,
          reportInterval=reportInterval)
    #
    return fitter

################# Bootstrap Analysis ##################
if os.path.isfile(PICKLE_FILE):
    fitters = pickle.load(open( PICKLE_FILE, "rb" ) )
else:
    # Construct fitters for all patients
    fitters = []
    for patient in patientDct.values():
        print("\n\n*** Processing patient %s""" % patient)
        fitter = mkFitter(patient)
        fitters.append(fitter)
        fitter.roadrunnerModel = None  # Cannot serialize this object
        pickle.dump(fitters, open( PICKLE_FILE, "wb" ) )

# Construct parameter plots
patients = [p+1 for p in patientDct.keys()]
fig, axes = plt.subplots(5,1, figsize=(12,10))
for pos, param in enumerate(PARAMS):
    ax = axes[pos]
    param = PARAMS[pos]
    means = [f.bootstrapResult.meanDct[param] for f in fitters]
    stds = [f.bootstrapResult.stdDct[param] for f in fitters]
    ax.errorbar(patients, means, yerr=stds)
    ax.set_title(param)
    if pos == len(PARAMS) - 1:
        ax.set_xlabel("Patient")
plt.suptitle("Bootstrap Parameter Estimates")
plt.show()
