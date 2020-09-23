#! /bin/bash
# Profiles codes that use SBstoat. $1 is the file or use the default benchmark.
PATH="tests/benchmarkModelFitter.py"
if [ $# -eq 1 ]; then
    PATH="$1"
fi
echo "Profiling file ${PATH}"
python -m ${PATH} > profile.csv
echo " ncalls  tottime  percall  cumtime  percall filename:lineno(function)" > result.csv
grep namedTimeseries profile.csv >> result.csv
grep timeseriesPlotter profile.csv >> result.csv
grep modelFitter profile.csv >> result.csv
grep "_helper" profile.csv >> result.csv
