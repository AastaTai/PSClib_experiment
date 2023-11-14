#!/bin/bash
set -ex

# for data in 10000 20000
# do
#     python experiments.py --methods sc kmeans psc --size ${data}
# done

for data in 30000 50000 70000 -1
do
    python experiments.py --methods kmeans psc --size ${data}
done
