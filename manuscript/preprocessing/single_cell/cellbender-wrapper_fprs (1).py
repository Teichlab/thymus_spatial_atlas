import scanpy as sc
import numpy as np
import sys
import os

# Perform Cellbender on a given sample, with proper cell count parameterisation
# and output post-processing to ensure importability via sc.read_10_h5()
#
# This version uses the cellranger 2 knee point evaluation to get the cell total
#
# Usage: python cellbender_cite_wrapper.py SAMPLE
# where SAMPLE is a folder with SAMPLE/cellranger_raw.h5

#this is required so cellbender can be called from within the thing
os.environ['MKL_THREADING_LAYER'] = 'GNU'

#the sample is the sole positional argument
sample = sys.argv[1]

#get the number of the barcodes in the cellranger 2 knee point evaluation
adata = sc.read_10x_h5(sample+"/cellranger_raw.h5")
#precomputes required QC metrics while not doing any actual filtering
sc.pp.filter_cells(adata, min_counts=0)
#get the top 3000 UMI counts, and then the 99th percentile of that
cell_counts = np.sort(adata.obs['n_counts'])[::-1][:3000]
umi_threshold = np.percentile(cell_counts, 99)
#count total cells with UMIs at 10+% of this level
filtered = np.sum(adata.obs['n_counts'] > umi_threshold/10)

#delete the adata as it's served its purpose
del adata

#run cellbender. this is all happening from within a job, so it's cool
#run with default FPR and then some really high CITE-minded ones
os.system("cellbender remove-background --input "+sample+"/cellranger_raw.h5 --output "+sample+"/"+sample+".h5 --expected-cells "+str(filtered)+" --total-droplets-included "+str(filtered+15000)+" --fpr 0.01 0.5 0.6 0.7 0.8 0.9 --cuda")