from SBstoat import NamedTimeseries, ModelFitter, SuiteFitter, \
      ObservationSynthesizerRandomErrors

import tellurium as te
import matplotlib
matplotlib.use('TkAgg')



# Model of a linear pathway
MODEL1 = """ 
# Reactions
    J1: $X0 -> S1; k1*$X0
    J2: S1 -> S2; k2*S1
    J3: S2 -> $S3; k3*S2
# Species initializations
    X0 = 10;
    S1 = 0;
    S2 = 0;
    S3 = X0
    k1 = 1;
    k2 = 2;
    k3 = 3;
"""
PARAMETERS1 = ["k1", "k2", "k3"]


# Model of a linear pathway
MODEL2 = """ 
# Reactions
    J1: $S2 -> S3; k3*$S2
    J2: S3 -> S4; k4*S3
    J3: S4 -> $X1; k5*S4
# Species initializations
    S2 = 10;
    S3 = 0;
    S4 = 0;
    X1 = S2;
    k3 = 3;
    k4 = 4;
    k5 = 5;
"""
PARAMETERS2 = ["k3", "k4", "k5"]


# To illustrate fitting suites of models, we create synthetic observational data based on the true model.
# There are two synthetic data sets, one for each of the experiments described above.


def mkSyntheticData(model, std=0.3):
    """
    Creates synthetic observations for a model by adding a normally distributed random variable with zero mean.
    
    Parameters
    ----------
    model: str
        Antimony model
    std: float
        Standard deviation of the random error
        
    Returns
    -------
        NamedTimeseries
    """
    rr = te.loada(model)
    dataArr = rr.simulate()
    fittedTS = NamedTimeseries(namedArray=dataArr)
    synthesizer = ObservationSynthesizerRandomErrors(
        fittedTS=fittedTS, std=std)
    return synthesizer.calculate()



data1TS = mkSyntheticData(MODEL1)
data2TS = mkSyntheticData(MODEL2)


# Fit the model suite
modelFitters = []
for model, dataTS, parametersToFit in  \
      zip([MODEL1, MODEL2], [data1TS, data2TS], [PARAMETERS1, PARAMETERS2]):
    modelFitter = ModelFitter(model, dataTS,
          parametersToFit=parametersToFit)
    modelFitters.append(modelFitter)
suiteFitter = SuiteFitter(modelFitters, modelNames=["Model1", "Model2"],
      fitterMethods=["differential_evolution"])
#suiteFitter.fitSuite()

numFold = 3
suiteFitter.crossValidate(numFold, isParallel=True)
print(suiteFitter.scoreDF)  # R^2 value for each fold
print(suiteFitter.parameterDF)  # Parameter estimates aggregated across folds
