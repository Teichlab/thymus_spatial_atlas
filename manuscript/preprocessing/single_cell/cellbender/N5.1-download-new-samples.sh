#!/bin/bash
set -eo pipefail

#some more samples to bend like in N3, with mappings located like so:

for SAMPLE in GEX{1..6} GEX{8..12}
do
	MAPPING="/lustre/scratch127/cellgen/cellgeni/tickets/tic-1740/new_starsolo/${SAMPLE}/output/Gene"
	mkdir $SAMPLE
	rsync -r $MAPPING/raw $SAMPLE
	rsync $MAPPING/filtered/barcodes.tsv.gz $SAMPLE
	gunzip $SAMPLE/barcodes.tsv.gz
done

for SAMPLE in TA1307242{7..9}
do
	MAPPING="/lustre/scratch127/cellgen/cellgeni/tickets/tic-1930/gex_results/${SAMPLE}/output/Gene"
	mkdir $SAMPLE
	rsync -r $MAPPING/raw $SAMPLE
	rsync $MAPPING/filtered/barcodes.tsv.gz $SAMPLE
	gunzip $SAMPLE/barcodes.tsv.gz
done
