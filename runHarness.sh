#!/bin/bash
# Runs the test harness, handling issues with memory constraints
NAME="mainTestHarness"
# Start fresh
sf.sh ${NAME}.pcl SBstoat/${NAME}.log
# Run repeatedly
for s in 1 200 400 600 800
  do
     let end=199+$s
     echo "**Models $s to $end"
     python SBstoat/mainTestHarness.py --firstModel $s --numModel 200 --useExistingData --useExistingLog 2> /tmp/mainTestHarness.out
  done
# Do the plot
python SBstoat/mainTestHarness.py --plot
