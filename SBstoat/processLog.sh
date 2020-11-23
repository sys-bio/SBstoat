#!/bin/bash
# Bash file to post-process a log file to determine the results of
# analyzing BioModels.

function report {
  count=`grep "$1" mainTestHarness.log | wc | awk {'print $1'}`
  echo $count
}


echo "Files processed: " `report "xml"`
echo "Successfully processed: " `report '\*[1-9][0-9]*.* bootstrap'`
echo "No fitable parameters: " `report 'No fitable parameters'`
echo "CVODE fails: " `report '(TestHarness failed.*CVODE'`
echo "Non-empty list: " `report 'Must provide a non-empty list'`
echo "SBML error: " `report 'TestHarness failed.*SBML error(s) when'`
