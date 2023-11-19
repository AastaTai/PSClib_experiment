#!/bin/bash
set -ex

for data in 10000 20000 30000 40000 50000 60000 -1
do
    python experiments.py --methods kmeans psc --size ${data}
done