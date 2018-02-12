#!/bin/bash

rm -rf build
./setup.py build
rc=$?
if [[ "$rc" ==  "0" ]]
then
    ./setup.py build_ext --inplace
    rc=$?
fi
exit $rc
