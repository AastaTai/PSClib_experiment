#!/bin/bash
set -ex

for data in 1000 3000 5000 7000 10000
do
    python experiments.py --methods sc kmeans psc --size ${data}
done

for data in -1
do
    python experiments.py --methods kmeans psc --size ${data}
done