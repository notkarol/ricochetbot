#!/bin/bash

rm -rf build
./setup.py build
./setup.py install --user
python3 -c 'import ricochet'
