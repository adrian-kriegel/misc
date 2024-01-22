#!/bin/sh

if [ -d "$EIGEN_INCLUDE_DIR" ]; then
    g++ -I $EIGEN_INCLUDE_DIR ./benchmark.cpp -o benchmark
else
    echo "Missing EIGEN_INCLUDE_DIR."
fi
