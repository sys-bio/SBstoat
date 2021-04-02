 <table style="width:100%">
  <tr>
    <td><img src="https://api.travis-ci.org/sys-bio/SBStoat.svg?branch=master" width="100"/></td>
    <td><img src="https://codecov.io/gh/sys-bio/SBStoat/branch/master/graph/badge.svg" /></td>
  </tr>
</table> 

![alt text](SBstoat_logo.png "")

 
<a href="https://codecov.io/gh/sys-bio/SBstoat">
</a>



# Fitting SBML Models with Tellurium

This repo provides easy-to-use tools for doing parameter fitting using the Tellurium simulator.
The project is named after the stoat, an animal that has superb skills at fitting into small places.
``SBstoat`` provides the following:

* Parameter fitting for a single model and for model suites (collections of models with overlapping parameters). The user can select any optimization method or combinations of methods that are available in ``scipy.optimize``.
* A variety of plots to assess the quality of parameter fits.
* Cross validation to assess model quality.
* Bootstrapping for estimating confidence intervals for parameters.
* Multiprocess implementation for increased performance and scaling.

A [Jupyter Notebook](https://github.com/sys-bio/SBstoat/blob/master/notebooks/Tutorial%20on%20Utilities%20for%20Model%20Fitting.ipynb) of detailed examples can be found here. Below is a summary.

The main module is `modelFitter`. A typically parameter fitting session is
shown below. For convenience, the model is expressed using the [Antimony](http://antimony.sourceforge.net/) modeling language.
However, SBML models can be loaded into [tellurium](http://tellurium.analogmachine.org/), and a tellurium object can be used in place of the antimony model.

    ANTIMONY_MODEL = """ 
    # Reactions   
    J1: S1 -> S2; k1*S1
    J2: S2 -> S3; k2*S2
    J3: S3 -> S4; k3*S3
    J4: S4 -> S5; k4*S4
    J5: S5 -> S6; k5*S5;
    # Species initializations
    S1 = 10; S2 = 0; S3 = 0; S4 = 0; S5 = 0; S6 = 0;
    k1 = 1; k2 = 2; k3 = 3; k4 = 4; k5 = 5;
    """
Now suppose we have the data file `tst_data.txt`. To fit this model to these data and see a report on the fit:

    # Fit parameters to ts1
    from SBstoat.modelFitter import ModelFitter
    fitter = ModelFitter(ANTIMONY_MODEL, "tst_data.txt", ["k1", "k2", "k3", "k4", "k5"])
    fitter.fitModel()
    print(fitter.reportFit())
    
The output is:

    [Fit Statistics]]
    # fitting method   = leastsq
    # function evals   = 49
    # data points      = 180
    # variables        = 5
    chi-square         = 73.2546170
    reduced chi-square = 0.41859781
    Akaike info crit   = -151.822803
    Bayesian info crit = -135.858019
    
    [[Variables]]
    k1:  0.95579053 +/- 0.03816186 (3.99%) (init = 5)
    k2:  2.24079567 +/- 0.19847112 (8.86%) (init = 5)
    k3:  2.96763525 +/- 0.35879852 (12.09%) (init = 5)
    k4:  3.07652723 +/- 0.39858904 (12.96%) (init = 5)
    k5:  5.90802238 +/- 1.43620318 (24.31%) (init = 5)
    
    [[Correlations]] (unreported correlations are < 0.100)
    C(k4, k5) = -0.248
    C(k3, k4) = -0.226
    C(k2, k3) = -0.218
    C(k3, k5) = -0.211
    C(k2, k4) = -0.189
    C(k1, k2) = -0.179
    C(k2, k5) = -0.178
    C(k1, k3) = -0.147
    C(k1, k5) = -0.144
    C(k1, k4) = -0.141
    
You can also get bootstrap estimates of parameter values. Because bootstrapping is computationally intensive, SBstoat uses multiple processes on your machine.

    # Get estimates of parameters
    fitter.bootstrap(numIteration=2000, reportInterval=500)
    fitter.reportBootstrap()
    
Here is the output:
    
    **Running bootstrap for 2000 iterations with 4 processes.
    bootstrap completed 500 iterations.
    bootstrap completed 1000 iterations.
    bootstrap completed 1500 iterations.
    Completed bootstrap process 2.
    Completed bootstrap process 3.
    Completed bootstrap process 4.
    bootstrap completed 2000 iterations.
    Completed bootstrap process 1.

    Bootstrap Report.
    Total iterations: 2000
    Total simulation: 2000
    k1
      mean: 0.9666458789599315
      std: 0.03984278523619386
      [2.5, 97.55] Percentiles: [0.89206257 1.04470717]
    k2
      mean: 2.1808554007110637
      std: 0.17819579282363782
      [2.5, 97.55] Percentiles: [1.85917689 2.56348925]
    k3
      mean: 3.233849345953018
      std: 0.4074066158009789
      [2.5, 97.55] Percentiles: [2.57874824 4.12921803]
    k4
      mean: 3.1037923601143054
      std: 0.38872479522475384
      [2.5, 97.55] Percentiles: [2.46792396 4.06937082]
    k5
      mean: 5.9285194938461565
      std: 1.0301263970600283
      [2.5, 97.55] Percentiles: [4.42373341 8.44386604]

More details of the features of `SBstoat` can be found in this
[tutorial](https://github.com/sys-bio/SBstoat/blob/master/notebooks/Tutorial%20on%20Utilities%20for%20Model%20Fitting.ipynb).

# Installation and validation
1. `pip install SBstoat`
1.  Verify the installation

    1. `git clone https://github.com/sys-bio/SBstoat.git`  to get the repository
    1. `cd SBstoat`
    1. `nosetests tests`

# Release Notes
## Release 1.14
* Support for suites of models. A suite is a collection of models with overlapping sets of parameters. A common use case is having model variants (e.g., different initial concentrations of floating species or gene knock-outs) that reflect different experimental conditions. Parameter fitting requires simultaneously fitting all models in the suite. See the class ``SuiteFitter``.
* Cross validation. Provides a way to assess model quality and estimates of parameter variance. Once you have an instance of ``ModelFitter``, invoke the method ``crossValidate(numFold)``, where ``numFold`` is the number of folds.
* Progress bar. Long running activities have a progress bar. In this release, only bootstrapping has a progress bar. Future releases will extend this.
* Random restarts for fitting. The quality of a fit often depends on the initial values used for parameters. The optional keyword ``numRestart`` for constructing ``ModelFitter`` indicates the number of random restarts to use in a fit.

## Release 1.16
* Benchmark for ``SuiteFitter``, ``benchmarkSuiteFitter.py``.
* Improved performance of SuiteFitter by a factor of 7.
* Parallel implementation of Cross Validation
* [Cross validation for ``SuiteFitter``.]


# Developer Notes

1. run tests as follows:
   1. change to this directory
   1. set the environment variable `PYTHONPATH` to
      the absolute path of this directory.
      - [Windows](https://www.computerhope.com/issues/ch000549.htm)
      - Linux and Mac
        - `PYTHONPATH=<current directory>`
        - `export PYTHONPATH`
   
   1. `nosetests tests`


