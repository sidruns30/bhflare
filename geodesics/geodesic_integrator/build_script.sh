#!/usr/bin/env bash
rm -r build
rm *.so
python setup.py build_ext --inplace
mv *.so ..
mv build/ ..