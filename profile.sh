# Profiles codes that use SBstoat. $1 is the file or use the default benchmark.
PPATH="tests/benchmarkModelFitter.py"
if [ $# -eq 1 ]; then
    PPATH="$1"
fi
echo "Profiling file ${PPATH}"
python -m cProfile ${PPATH} > profile.csv
exit()
echo " ncalls  tottime  percall  cumtime  percall filename:lineno(function)" > result.csv
grep "^  *[1-9].*:" profile.csv >> result.csv
sed 's/ /,/g' result.csv | sed 's/,,/,/g' | sed 's/,,/,/g' | sed 's/,,/,/g' | sed 's/^,//' > result.csv
