# Influenza, SARS and SARS-CoV2 in Tellurium

### A simple target cell-limited model, T ==> E ==> I --> produce V:
# T - number of target cells
# E - number of exposed cells (virus replicating inside, not yet spreading virus)
# I - number of infected cells (active virus production)
# V - viral titre, in units of TCID50/ml of biofluid wash (for Influenza)
	
### The ODEs
# dT/dt = - beta*T*V
# dE/dt =   beta*T*V - kappa*E;
# dI/dt =   kappa * E - delta*I;
# dV/dt =   p*y(I) - c*y(V);


import tellurium as te
import SBstoat



ANTIMONY_MODEL  = '''
    // Equations
    E1: T -> E ; beta*T*V ; // Target cells to exposed
    E2: E -> I ; kappa*E ;  // Exposed cells to infected
    E3: -> V ; p*I ; 	    // Virus production by infected cells
    E4: V -> ; c*V ; 	    // Virus clearance
    E5: I -> ; delta*I 	    // Death of infected cells    

    // Parameters - from the Influenza article,
        
    beta = 3.2e-5;  // rate of transition of target(T) to exposed(E) cells, in units of 1/[V] * 1/day
    kappa = 4.0;    // rate of transition from exposed(E) to infected(I) cells, in units of 1/day
    delta = 5.2;    // rate of death of infected cells(I), in units of 1/day
    p = 4.6e-2;     // rate virus(V) producion by infected cells(I), in units of [V]/day
    c = 5.2; 	    // rate of virus clearance, in units of 1/day

    // Initial conditions
    T = 4E+8 // estimate of the total number of susceptible epithelial cells
             // in upper respiratory tract)
    E = 0
    I = 0
    V = 0.75 // the dose of virus in TCID50 in Influenza experiment; could be V=0 and I = 20 instead for a natural infection

'''


# V vs time (days)


# data section ============================================

# All viral data is in log10(load),...
# log10(load predicted by model) may be needed for data fitting


#Influenza.csv
# Influenza A data - 5 patients
# viral levels in log10(TCID50 / ml of nasal wash)
# time in days since volunteer exposure
# each line in the array is an individual volunteer



#SARS_CoV2_sputum.csv and SARS_CoV2_nasal.csv
# SARS-CoV-2 data - 9 patients,
# for each patient - viral loads from lungs (sputum) and from nasal cavity (swab)
# viral levels in log10(RNA copies / ml sputum), ...
# respectively log10(RNA copies / nasal swab)
# time in days since symptoms onset
# corresponding lines in the two arrays belong to an individual patient



#SARS.csv
# SARS data recorded from 12 patients;
# included them just for comparison, probably too few datapoints for model inference
# viral levels in log10(RNA copies / ml of nasopharingeal aspirate)
# time - only three samples per patient, at 5, 10 and 15 days post symptoms onset

# Fit parameters to ts1
from SBstoat.modelFitter import ModelFitter
fitter = ModelFitter(ANTIMONY_MODEL, "Influenza-1.txt", ["beta","kappa","delta","p","c"])
fitter.fitModel()
print(fitter.reportFit())

fitter.plotFitAll(numRow=2, numCol=2)
fitter.plotResiduals(numRow=2, numCol=2)


# Get estimates of parameters
fitter.bootstrap(numIteration=2000, reportInterval=500)
print(fitter.getFittedParameterStds())





