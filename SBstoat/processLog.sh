#!/bin/bash
# Bash file to post-process a log file to determine the results of
# analyzing BioModels.

TEMP="/tmp/processLog.log"
INPUT="mainTestHarness.log"

function report {
  count=`grep "$1" ${TEMP} | wc | awk {'print $1'}`
  echo $count
}

# Pre-process the log to remove duplicate lines for models processed
# This doesn't work because it doesn't handle duplicates in outcomes
#cat ${INPUT} | sed '/\*Model.*xml/s/^.*BIO/BIO/' | sort | uniq -u > ${TEMP}
cp ${INPUT} ${TEMP}
# Analyze the log
echo "Files processed: " `report "xml"`
echo "Successfully processed: " `report '\*[1-9][0-9]*.* bootstrap'`
echo "No bootstrap results: " `report '\*0 *.* bootstrap'`
echo "No fitable parameters: " `report 'No fitable parameters'`
echo "CVODE fails: " `report '(TestHarness failed.*CVODE'`
echo "Non-empty list: " `report 'Must provide a non-empty list'`
echo "SBML error: " `report 'TestHarness failed.*SBML error(s) when'`
echo "min == max: " `report 'TestHarness failed.*min == max'`
echo "Bad path: " `report 'sbmlPath is not a valid'`
