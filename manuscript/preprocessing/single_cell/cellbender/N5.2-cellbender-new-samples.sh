#!/bin/bash
set -eo pipefail

#bend some new samples, N3 style

for SAMPLE in GEX{1..6} GEX{8..12} TA1307242{7..9}
do
    #between N3 and now, the GPU queues got a lot more contested
    #use gentler bsub syntax so jobs go out easier
    #use a LOT of RAM as these are heavy somehow
    bsub -J cb -o 'logs/cb.%J.out' -e 'logs/cb.%J.err' -q gpu-normal -n 1 -M 50000 -R"select[mem>50000] rusage[mem=50000] span[hosts=1]" -gpu"mode=shared:j_exclusive=no:gmem=10000:num=1" "python cellbender-wrapper.py $SAMPLE"
done
