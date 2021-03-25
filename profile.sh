# Profiles codes that use SBstoat. $1 is the file or use the default benchmark.
PPATH="tests/benchmarkModelFitter.py"
if [ $# -eq 1 ]; then
    PPATH="$1"
fi
echo "Profiling file ${PPATH}"
python -m cProfile ${PPATH} > profile1.csv
echo " ncalls  tottime  percall  cumtime  percall filename:lineno(function)" > profile.csv
grep "^  *[1-9].*:" profile1.csv >> profile.csv
sed 's/ /,/g' profile.csv | sed 's/,,/,/g' | sed 's/,,/,/g' | sed 's/,,/,/g' | sed 's/^,//' > profile2.csv
cp profile2.csv profile.csv
mv profile1.csv /tmp
mv profile2.csv /tmp
