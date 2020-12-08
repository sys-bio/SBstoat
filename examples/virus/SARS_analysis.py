#!/usr/bin/env python
# coding: utf-8

# # Fitting Parameters for SARS Data

# ## Overview
# 
# This is a study COVID data. The analysis provides plots of fits and parameter estimates. 
# 
# ### Data (CoV_nasal.csv)
# 
# - 9 patients (columns)
# 
# ### State variables
# 
# - $T$: number of target cells
# - $E$: number of exposed cells (virus replicating inside, not yet spreading virus)
# - $I$: number of infected cells (active virus production)
# - $V$: viral titre, in units of TCID50/ml of biofluid wash (for Influenza)
# 
# ### Model: $T \rightarrow E \rightarrow I \rightarrow \emptyset$
#  $\frac{dT}{dt} = - \beta T V$
#  
#  $\frac{dE}{dt} =  \beta T V - \kappa E$
#  
#  $\frac{dI}{dt} = \kappa E - \delta I$
#  
#  $\frac{dV}{dt} = p y(I) - c y(V)$

print("# In[1]:")


# Python packages used
import os
import numpy as np
import pandas as pd
import SBstoat
from SBstoat.modelStudy import ModelStudy
import matplotlib
matplotlib.use('TkAgg')


print("# In[2]:")


# Programming Constants Used in Analysis. Constants are in all capital letters.
USE_SERIALIZED = False  # Use saved values of fitting from a previous bootstrap (if present)
DO_SERIALIZE = False  # Update the saved values of fitted data
DIR = "/home/ubuntu/SBstoat/examples/virus"  # Directory where the data are
FILE_NAME = "SARS_CoV2_nasal.csv"  # Name of the file containing the observed data
NUM_BOOTSTRAP_ITERATION = 1000  # Number of bootstrap iterations, if bootstrapping is done
VIRUS = "log10V"  # Name of the state variable that corresponds to the observed data


# ## Study for Baseline Model

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
    
    // Computed values
    log10V := log10(V)

'''


# ### 1. Data Setup`
# 
# The rows are patients; the columns are times. We need to create separate data for each patient.

print("# In[4]:")


# Transform the input data into separate data sources.
path = os.path.join(DIR, FILE_NAME)
patients = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9"]
dataSourceDct = SBstoat.modelStudy.mkDataSourceDct(path, VIRUS,
                                                   dataSourceNames=patients, isTimeColumns=True)


print("# In[5]:")


# dataSourceDct is a python dictionary. The key is 'Pn', where n is the patient number.
# The value is a time series for that patient.
dataSourceDct


# ### 2. Transform the simulation results to units of observed values
# The observed values are in units of log10. So, simulation results must
# be converted to these units. This is done by using an assignment rule in the simulation model.
# For this model, the assignmnt rule is ``log10V := log10(V)``.

# ### 3. Specify permissible values for parameters
# For each parameter, provide a tuple of its: lower bound, upper bound, and starting value.

print("# In[6]:")


# Parameter value ranges: lower, upper, initial value
parameterDct = dict(
      beta=(0, 10e-5, 3.2e-5),
      kappa=(0, 10, 4.0),
      delta=(0, 10, 5.2),
      p=(0, 1, 4.6e-2),
      c=(0, 10, 5.2)
      )


# ### 4. Run the model and produce plots.

print("# In[7]:")


# Run a study
def runStudy(model, dirStudyPath):
    study = ModelStudy(model,                     # Antimony model to evaluate
                   dataSourceDct,                 # Data sources to use for fitting
                   parameterDct=parameterDct,     # Parameters and their value ranges
                   dirStudyPath=dirStudyPath,     # Where to store the results of bootstrapping
                   selectedColumns=["log10V"],    # Output column is computed in the assignment rule
                   doSerialize=DO_SERIALIZE,      # Save the results of bootstrapping
                   useSerialized=USE_SERIALIZED)  # Use previously calculated bootstrap results if they are present

    study.bootstrap(numIteration=NUM_BOOTSTRAP_ITERATION)  # Do bootstrapping
    print("\n\n")
    study.plotFitAll(ylim=[0, 9])                          # Plot fitted and observed values with band plots for confidence
    print("\n\n")
    study.plotParameterEstimates()                         # Plot the parameter estimates for each data source


print("# In[8]:")


dirStudyPath = os.path.join(DIR, "SARS_StudyFitters_01")
runStudy(ANTIMONY_MODEL, dirStudyPath)

