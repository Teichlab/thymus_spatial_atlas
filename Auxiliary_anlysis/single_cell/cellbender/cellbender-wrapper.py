import tables
import numpy as np
import sys
import os

# Perform Cellbender on a given sample, with proper cell count parameterisation
# and output post-processing to ensure importability via sc.read_10_h5()
#
# Usage: python cellbender-wrapper.py SAMPLE
# where SAMPLE is a folder with SAMPLE/raw, and SAMPLE/barcodes.tsv (filtered barcodes)

#this is required so cellbender can be called from within the thing
os.environ['MKL_THREADING_LAYER'] = 'GNU'

#the sample is the sole positional argument
sample = sys.argv[1]

#we just need the number of these barcodes
#but it's easiest/most consistent between different mapping tools
#to ask for the list rather than look for QC files
with open(sample+"/barcodes.tsv", "r") as fid:
    filtered = len(fid.readlines())

#run cellbender. this is all happening from within a job, so it's cool
os.system("cellbender remove-background --input "+sample+"/raw --output "+sample+"/"+sample+".h5 --expected-cells "+str(filtered)+" --total-droplets-included "+str(filtered+15000)+" --cuda")

#fix the cellbender output files - SAMPLE/SAMPLE.h5 and SAMPLE/SAMPLE_filtered.h5
#after this operation they should be importable via scanpy
for h5_to_fix in [".h5", "_filtered.h5"]:
    with tables.open_file(sample+"/"+sample+h5_to_fix, "r+") as f:
        n = f.get_node("/matrix/features")
        n_genes = f.get_node("/matrix/shape")[0]
        if "genome" not in n:
            f.create_array(n, "genome", np.repeat("GRCh38", n_genes))
