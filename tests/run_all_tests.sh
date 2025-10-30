#!/bin/bash

# Provide directory for tests 
IONSIM_TESTS_DIR="/Users/ecmcgar/ionsim/tests"

IONSIM_Venv="/Users/ecmcgar/ionsim_env/bin"

source "$IONSIM_Venv"/activate

if [ ! -d "$IONSIM_TESTS_DIR" ]; then
  echo Error: Directory $IONSIM_TESTS_DIR does not exist.
  exit 1
fi 

# Collect test files 
test_files=("$IONSIM_TESTS_DIR"/test_*.py)

# Check if there are test files 
if [ ${#test_files[@]} -eq 0 ]; then
  echo No test files found in '$IONSIM_TESTS_DIR'.
  exit 0
fi 

# Run each test: 
for test_file in "${test_files[@]}"; do
  echo Running test: "$test_file"
  python3 "$test_file"
done 


#echo All tests ran successfully!
