#!/bin/bash
set -eo pipefail

#prior to running this script, source activate cellbender
#(to wake the right conda environment)

#the stuff lives in this directory... in five subdirectories. fun!
#the command extracts the master list of samples across all the subfolders
for SAMPLE in `ls /nfs/team205/vk8/irods_data/09_thymus/starsolo/*/*/output/Gene | grep "starsolo" | cut -f 9 -d "/"`
do
	mkdir $SAMPLE
	#this will be able to locate the relevant raw count matrix and copy it over
	cp -r /nfs/team205/vk8/irods_data/09_thymus/starsolo/*/$SAMPLE/output/Gene/raw $SAMPLE
	~/jsub lsf -n pan -q gpu-normal -c 1 -m 5g "cellbender remove-background --input $SAMPLE/raw --output $SAMPLE/$SAMPLE.h5 --cuda" | bsub -G team269
done
