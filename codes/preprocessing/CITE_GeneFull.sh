#!/bin/bash
set -eo pipefail

#start by copying in the sample data
#this has to be done pre job as as jobs don't see /warehouse

for SAMPLE in GEX12
do
    mkdir $SAMPLE
    cp -r /lustre/scratch117/cellgen/cellgeni/TIC-starsolo/tic-1740/starsolo_results/$SAMPLE/output/GeneFull/raw  $SAMPLE
    cp /lustre/scratch117/cellgen/cellgeni/TIC-starsolo/tic-1740/starsolo_results/$SAMPLE/output/GeneFull/filtered/barcodes.tsv.gz $SAMPLE
    gunzip $SAMPLE/barcodes.tsv.gz
done

#run cellbender pipeline
for SAMPLE in GEX12
do
    ~/jsub lsf -n pan -q gpu-normal -c 1 -m 5g "python cellbender-wrapper.py $SAMPLE" | bsub -G teichlab
done
