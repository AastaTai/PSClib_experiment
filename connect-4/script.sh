#!/bin/bash
set -ex

for data in 10000 30000 50000 -1
do
    python experiments.py --methods kmeans psc --size ${data}
done