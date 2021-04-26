import tellurium as te
import SBstoat
import time
from SBstoat.namedTimeseries import NamedTimeseries
from SBstoat.suiteFitter import SuiteFitter
from copy import deepcopy

then = time.time()

modelStrFront = '''
function Fiii(ri1, ri2, ri3, kf, kr, i1, i2, i3, s, p, Kmi1, Kmi2, Kmi3, Kms, Kmp, wi1, wi2, wi3, ms, mp)
    ((ri1+(1-ri1)*(1/(1+i1/Kmi1)))^wi1) * ((ri2+(1-ri2)*(1/(1+i2/Kmi2)))^wi2) * ((ri3+(1-ri3)*(1/(1+i3/Kmi3)))^wi3) * (kf*(s/Kms)^ms)/((s/Kms)^ms+1)-kr*(p/Kmp)^mp
end
function Fi(ri, kf, kr, i, s, p, Kmi, Kms, Kmp, wi, ms, mp)
    ((ri+(1-ri)*(1/(1+i/Kmi)))^wi)*(kf*(s/Kms)^ms)/((s/Kms)^ms+1)-kr*(p/Kmp)^mp
end

function Fss(kf1, kf2, kr, s1, s2, p, Kms1, Kms2, Kmp, ms1, ms2, mp)
    (kf1*(s1/Kms1)^ms1)*(kf2*(s2/Kms2)^ms2)/(((s1/Kms1)^ms1+1)*((s2/Kms2)^ms2+1))-kr*(p/Kmp)^mp
end

function F0(kf, kr, s, p, Kms, Kmp, ms, mp)
    (kf*(s/Kms)^ms)/((s/Kms)^ms+1)-kr*(p/Kmp)^mp
end
function Ha(V, K, n, K2, n2, t, s)
    ((s^n2)/((s^n2) + (K2^n2))) * ((V * n * (K^n) * (t^(n-1))) / (((K^n) + (t^n))^2))
end
function Hi(V, K, n, K2, n2, t, s)
    (1/((s^n2) + (K2^n2))) * ((V * n * (K^n) * (t^(n-1))) / (((K^n) + (t^n))^2))
end

model coarseEGF_0_0()

// Reactions

FreeLigand: Lp -> L; Fss(kf_01, kf_02, kr_0, Lp, E, L, Kms_01, Kms_02, Kmp_0, ms_01, ms_02, mp_0);

Phosphotyrosine: -> P; Fi(ri_0, kf_1, kr_1, Mig6, L, P, Kmi_1, Kms_1, Kmp_1, wi_1, ms_1, mp_1);
Ras: -> R; Fiii(ri1_1, ri2_1, ri3_1, kf_2, kr_2, Spry2, P, E, P, R, Kmi1_2, Kmi2_2, Kmi3_2, Kms_2, Kmp_2, wi1_2, wi2_2, wi3_2, ms_2, mp_2);
Erk: -> E; F0(kf_3, kr_3, R, E, Kms_3, Kmp_3, ms_3, mp_3);
Precursor: -> Lp; kLp*C1;
Spry: -> Spry2; Ha(Va, Ka, na, Ka2, na2, t, L);
Mig: -> Mig6; Hi(Vi, Ki, ni, Ki2, ni2, t, L);

// Species IVs
'''

modelStrBack = '''
P = 0;
R = 0;
E = 0;
Lp = 0;
Mig6 = 0.325;
Spry2 = 3.854;
C1 = 1;

// Parameter values
kf_01 = 1;
kf_02 = 1;
kr_0 = 1;
Kms_01 = 1;
Kms_02 = 1;
Kmp_0 = 1;
ms_01 = 1;
ms_02 = 1;
mp_0 = 1;
ri_0 = 1;
kf_1 = 1;
kr_1 = 1;
Kmi_1 = 1;
Kms_1 = 1;
Kmp_1 = 1;
wi_1 = 1;
ms_1 = 1;
mp_1 = 1;
ri1_1 = 1;
ri2_1 = 1;
ri3_1 = 1;
kf_2 = 1;
kr_2 = 1;
Kmi1_2 = 1;
Kmi2_2 = 1;
Kmi3_2 = 1;
Kms_2 = 1;
Kmp_2 = 1;
wi1_2 = 1;
wi2_2 = 1;
wi3_2 = 1;
ms_2 = 1;
mp_2 = 1;
kf_3 = 1;
kr_3 = 1;
Kms_3 = 1;
Kmp_3 = 1;
ms_3 = 1;
mp_3 = 1;

t := time;

kLp = 0.00383;
Va = 23.0314252;
Ka = 10153.3423;
na = 2.52290896;
Ka2 = 20.134033969;
na2 = 8.48874275;
Vi = 0.790702084;
Ki = 8461.68691;
ni = 1.49168606;
Ki2 = 0.160718999;
ni2 = 0.975509826;

end
'''

ligandStrs = ['L = 0;', 'L = 0.00167;', 'L = 0.005;', 'L = 0.0167;', 'L = 0.05;',
              'L = 0.167;', 'L = 0.5;', 'L = 1.67;', 'L = 5;', 'L = 16.7;']

model1 = te.loada(modelStrFront + ligandStrs[0] + modelStrBack)
model2 = te.loada(modelStrFront + ligandStrs[1] + modelStrBack)
model3 = te.loada(modelStrFront + ligandStrs[2] + modelStrBack)
model4 = te.loada(modelStrFront + ligandStrs[3] + modelStrBack)
model5 = te.loada(modelStrFront + ligandStrs[4] + modelStrBack)
model6 = te.loada(modelStrFront + ligandStrs[5] + modelStrBack)
model7 = te.loada(modelStrFront + ligandStrs[6] + modelStrBack)
model8 = te.loada(modelStrFront + ligandStrs[7] + modelStrBack)
model9 = te.loada(modelStrFront + ligandStrs[8] + modelStrBack)
model10 = te.loada(modelStrFront + ligandStrs[9] + modelStrBack)

paramR = SBstoat.Parameter('R', lower=0, upper=3, value=2.5)
paramE = SBstoat.Parameter('E', lower=0, upper=11, value=10)
paramP = SBstoat.Parameter('P', lower=0, upper=10, value=0)
paramLp = SBstoat.Parameter('Lp', lower=0, upper=10, value=0)

params1 = [paramR, paramE, paramP, paramLp, "kf_01", "kf_02", "kr_0", 
#          "Kma_01", "Kma_02",  # these aren't in the model
           "Kmp_0", "ms_01", "ms_02", "mp_0", "ri_0", "kf_1", "kr_1",
           "Kmi_1", "Kms_1", "Kmp_1", "wi_1", "ms_1", "mp_1", "ri1_1", "ri2_1",
           "ri3_1", "kf_2", "kr_2", "Kmi1_2", "Kmi2_2", "Kmi3_2", "Kms_2",
           "Kmp_2", "wi1_2", "wi2_2", "wi3_2", "ms_2", "mp_2", "kf_3", "kr_3",
           "Kms_3", "Kmp_3", "ms_3", "mp_3"]

params2 = deepcopy(params1)
params3 = deepcopy(params1)
params4 = deepcopy(params1)
params5 = deepcopy(params1)
params6 = deepcopy(params1)
params7 = deepcopy(params1)
params8 = deepcopy(params1)
params9 = deepcopy(params1)
params10 = deepcopy(params1)

# optimizerMethod = SBstoat.OptimizerMethod(method="differential_evolution",
#                                           kwargs={"popsize": 10, "tol": 0.000001, 'max_nfev': 1000000})

optimizerMethod = SBstoat.OptimizerMethod(method="differential_evolution",
                                          kwargs={"popsize": 10, 'max_nfev': 1000000})

modelFitters = []

items = zip([model1, model2, model3, model4, model5, model6, model7, model8, model9, model10],
            ["egf1.csv", "egf2.csv", "egf3.csv", "egf4.csv", "egf5.csv",
             "egf6.csv", "egf7.csv", "egf8.csv", "egf9.csv", "egf10.csv"],
            [params1, params2, params3, params4, params5, params6, params7, params8, params9, params10])

for model, data, parametersToFit in items:
    modelFitter = SBstoat.ModelFitter(model, data, parametersToFit=parametersToFit)
    modelFitters.append(modelFitter)

suiteFitter = SuiteFitter(modelFitters, modelNames=["Model 1", "Model 2", "Model 3", "Model 4", "Model 5",
                                                      "Model 6", "Model 7", "Model 8", "Model 9", "Model 10"],
                            fitterMethods=[optimizerMethod])

suiteFitter.fitSuite()
print(suiteFitter.reportFit())

now = time.time()
print()
print(now - then)
