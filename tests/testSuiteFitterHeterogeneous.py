# Test for hetergenous collection of fitters

from SBstoat import NamedTimeseries, ModelFitter, SuiteFitter, \
      ObservationSynthesizerRandomErrors
import SBstoat._constants as cn

import tellurium as te
import unittest


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

DATA1_TS = mkSyntheticData(MODEL1)
DATA2_TS = mkSyntheticData(MODEL2)
MODEL_COL = [MODEL1, MODEL2]
DATA_COL = [DATA1_TS, DATA2_TS]
PARAMETER_COL = [PARAMETERS1, PARAMETERS2]
MODEL_NAMES = ["Model1", "Model2"]


class TestSuiteFitter(unittest.TestCase):

    def setUp(self):
        modelFitters = []
        items = zip(MODEL_COL, DATA_COL, PARAMETER_COL)
        for model, dataTS, parametersToFit in items:
            modelFitter = ModelFitter(model, dataTS,
                  parametersToFit=parametersToFit)
            modelFitters.append(modelFitter)
        self.suiteFitter = SuiteFitter(modelFitters, modelNames=MODEL_NAMES,
              fitterMethods=["differential_evolution"])

    def testHeterogeneous(self):
        numFold = 3
        self.suiteFitter.crossValidate(numFold, isParallel=True)
        trues = [r2 > 0.9 for r2 in self.suiteFitter.scoreDF[cn.SCORE]]
        

if __name__ == '__main__':
    unittest.main()
