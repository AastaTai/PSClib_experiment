#!/bin/bash
set -ex

for data in 30000 50000 70000
do
    python experiments.py --methods kmeans psc --size ${data}
done