#!/bin/bash

rm -rf build
./setup.py build
rc=$?
if [[ "$rc" ==  "0" ]]
then
    ./setup.py install --user
    rc=$?
fi
exit $rc
