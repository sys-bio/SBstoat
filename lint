#! /bin/bash
# Runs pylint, eliminating some errors
 pylint $1 | \
  grep -v "name doesn't conform" |  \
  grep -v "missing docstring" |  \
  grep -v "third party import" |  \
  grep -v "standard import" | \
  grep -v "doesn't conform to snake"  | \
  grep -v "issing class docstring" | \
  grep -v "Too many instance attributes" | \
  grep -v "Too many local variables" | \
  grep -v "Too few public methods" | \
  grep -v "Too many arguments" | \
  grep -v "Too many statements" | \
  grep -v "Too many branches" | \
  grep -v "Access to a protected member" | \
  grep -v "Missing function or method docstring" | \
  grep -v "Catching too general exception Exception"
