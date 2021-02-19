
import tellurium as te
from SBstoat.modelFitter import ModelFitter
import matplotlib
matplotlib.use('TkAgg')

model = te.loada('''

function HillTime(V, K, n, t)
    ((V * n * (K^n) * (t^(n-1))) / (((K^n) + (t^n))^2))
end

model modular_EGFR_current_128()

// Reactions

SproutyFunc: -> Spry2; HillTime(V_0, K_0, n_0, t)


// Species IVs
Spry2 = 0;

// Parameter values
V_0 = 19.9059673;
K_0 = 10153.3568;
n_0 = 2.52290790;
t := time

end
''')

# sim = model.simulate(0, 7200, 7201)
# model.plot()
# quit()



fitter = ModelFitter(model, "spry2_2a.txt", ["V_0", "K_0", "n_0"],
                     fitterMethods='differential_evolution', parameterDct={
            "V_0": (10, 20, 40), "K_0": (1800, 6000, 20000), "n_0": (1, 2, 12)})
fitter.fitModel()
print(fitter.reportFit())
