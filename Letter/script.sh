#!/bin/bash
set -ex

for data in 5000 10000 -1
do
    python experiments.py --methods kmeans psc --size ${data}
done