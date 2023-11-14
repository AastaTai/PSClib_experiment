#!/bin/bash
set -ex

for data in 1000 3000 5000 7000 -1
do
    python experiments.py --methods sc kmeans psc --size ${data}
done