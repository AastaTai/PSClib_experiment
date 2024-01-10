#!/bin/bash
# set -ex
# for data in 1000 3000 5000
# do
#     python experiments.py --methods sc kmeans psc --size ${data}
# done
for data in 10000 20000 30000 40000 50000 -1
do
    python Firework/experiments.py --methods kmeans psc --size ${data}
done
