#!/bin/bash
# Runs the test harness, handling issues with memory constraints
NAME="mainTestHarness"
# Start fresh
sf.sh ${NAME}.pcl SBstoat/${NAME}.log
# Run repeatedly
for s in 1 200 400 600 800
  do
     python SBstoat/mainTestHarness.py --firstModel $s --numModel 200 --useExistingData --useExistingLog
  done
# Do the plot
python SBstoat/mainTestHarness.py --plot
