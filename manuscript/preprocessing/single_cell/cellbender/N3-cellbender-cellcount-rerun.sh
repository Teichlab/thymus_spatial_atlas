#!/bin/bash
set -eo pipefail

#prior to running this script, source activate cellbender
#(to wake the right conda environment)

#between N2 and N3, the filtered barcodes were downloaded and unzipped to SAMPLE/barcodes.tsv

#the stuff lives in this directory... in five subdirectories. fun!
#the command extracts the master list of samples across all the subfolders
for SAMPLE in `ls /nfs/team205/vk8/irods_data/09_thymus/starsolo/*/*/output/Gene | grep "starsolo" | cut -f 9 -d "/"`
do
    #use the newly made cellbender wrapper
    ~/jsub lsf -n pan -q gpu-normal -c 1 -m 5g "python cellbender-wrapper.py $SAMPLE" | bsub -G team269
done

#two of these refused to work with the cell settings, so just run them the old-fashioned way
for SAMPLE in TA1230344{7..8} TA12303451
do
    ~/jsub lsf -n pan -q gpu-normal -c 1 -m 5g "python cellbender-wrapper.py $SAMPLE" | bsub -G team269
done
for SAMPLE in TA123034{49..50}
do
    ~/jsub lsf -n pan -q gpu-normal -c 1 -m 5g "cellbender remove-background --input $SAMPLE/raw --output $SAMPLE/$SAMPLE.h5 --cuda" | bsub -G team269
done
