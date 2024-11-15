#!/bin/bash
set -eo pipefail

#prior to running this script, source activate cellbender
#(to wake the right conda environment)

for SAMPLE in TA123034{47..51}
do
	mkdir $SAMPLE
	cp -r /nfs/team205/vk8/irods_data/09_thymus/starsolo/tic-1407/$SAMPLE/output/Gene/raw $SAMPLE
	~/jsub lsf -n pan -q gpu-normal -c 1 -m 5g "cellbender remove-background --input $SAMPLE/raw --output $SAMPLE/$SAMPLE.h5 --cuda" | bsub -G team269
done
