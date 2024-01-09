#!/bin/bash
set -ex

for data in 15000 30000 45000
do
    python exp_jss.py --methods kmeans psc --size ${data}
done