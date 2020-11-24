#!/bin/bash
# Runs the test harness, handling issues with memory constraints
NAME="mainTestHarness"
# Start fresh
mv ${NAME}.pcl /tmp
mv SBstoat/${NAME}.log /tmp
# Run repeatedly
for s in 1 200 400 600 800
  do
     python SBstoat/mainTestHarness.py --firstModel $s --numModel 800 --useExistingData True --useExistingLog True
  done
