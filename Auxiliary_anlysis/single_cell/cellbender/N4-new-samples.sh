#!/bin/bash
set -eo pipefail

#start by copying in the sample data
#this has to be done pre job as as jobs don't see /warehouse

#for SAMPLE in TA9306459 TA9306460 TA9306461 TA10226224 TA10226225 TA10226226 TA10226227 TA10279111 TA10279112
#do
#    mkdir $SAMPLE
#    cp -r /warehouse/cellgeni/tic-1441/$SAMPLE/output/Gene/raw $SAMPLE
#    cp /warehouse/cellgeni/tic-1441/$SAMPLE/output/Gene/filtered/barcodes.tsv.gz $SAMPLE
#    gunzip $SAMPLE/barcodes.tsv.gz
#done

#run cellbender pipeline
for SAMPLE in TA9306459 TA9306460 TA9306461 TA10226224 TA10226225 TA10226226 TA10226227 TA10279111 TA10279112
do
    ~/jsub lsf -n pan -q gpu-normal -c 1 -m 5g "python cellbender-wrapper.py $SAMPLE" | bsub -G team269
done
