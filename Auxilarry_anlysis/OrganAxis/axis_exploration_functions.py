# linear regression of 2 axes with batch correction

import statsmodels.api as sm

def multi_linear_regression_all_genes(adata, factor1, factor2, batch):
    # Initialize a dictionary to hold results
    results = {}

    # Loop over all genes
    for gene in adata.var_names:
        # Extract the expression values for this gene
        gene_expression = adata[:, gene].X.toarray().flatten()

        # Create a mask for non-zero expression values
        mask = gene_expression > 0

        # Filter expression values and factor values
        y = gene_expression[mask]
        X = adata.obs.loc[mask, [factor1, factor2, batch]]

        # Add a constant to the covariate matrix
        X = sm.add_constant(X)

        # Fit the model using robust regression
        model = sm.RLM(y, X,M=sm.robust.norms.TrimmedMean(c=0.5))
        fitted = model.fit()

        # Save the results
        results[gene] = fitted

    return results

# Let's say 'manual_bin_cma_v2' is your cluster annotation
cluster_annotation = 'manual_bin_cma_v2'
adata_ibex_50.raw = adata_ibex_50
from sklearn.preprocessing import StandardScaler



# Make a dataframe with the mean values of the genes
cluster_means = pd.DataFrame({cluster: adata_ibex_50.raw.X[adata_ibex_50.obs[cluster_annotation] == cluster].mean(axis=0) for cluster in adata_ibex_50.obs[cluster_annotation].cat.categories},
                             index=adata_ibex_50.raw.var_names)

# Standardize the gene expression values for each gene across clusters
cluster_means_std = cluster_means.apply(lambda x: (x - x.mean()) / x.std(), axis=1)
cluster_means_std = cluster_means_std.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=1)

# Define the order of the clusters
ct_order = ['Capsular', 'Sub-Capsular', 'Cortical level I', 'Cortical level II', 'Cortical level III', 'Cortical CMJ', 'Medullar CMJ', 'Medullar level I', 'Medullar level II', 'Medullar level III', 'Medullar Deep']

# Reorder the dataframe according to the defined cluster order and transpose it
cluster_means_std = cluster_means_std[ct_order].transpose()

# Define the number of clusters and genes
num_clusters = len(cluster_means_std.index)
num_genes = len(cluster_means_std.columns)

# Set figure size
figsize = (2*num_clusters/0.5, 2*num_genes/10)  # Adjust the denominator to get the aspect ratio you want

# Perform hierarchical clustering and plot a heatmap
g = sns.clustermap(cluster_means_std, method='ward', metric='euclidean', cmap='viridis', row_cluster=False, figsize=figsize)
plt.setp(g.ax_heatmap.get_xticklabels()) 

plt.show()

# Save the plot
g.figure.savefig("bin_heatmap.pdf")


# Cluster binned axis 

# Let's say 'manual_bin_cma_v2' is your cluster annotation
cluster_annotation = 'manual_bin_cma_v2'
adata_ibex_50.raw = adata_ibex_50
from sklearn.preprocessing import StandardScaler



# Make a dataframe with the mean values of the genes
cluster_means = pd.DataFrame({cluster: adata_ibex_50.raw.X[adata_ibex_50.obs[cluster_annotation] == cluster].mean(axis=0) for cluster in adata_ibex_50.obs[cluster_annotation].cat.categories},
                             index=adata_ibex_50.raw.var_names)

# Standardize the gene expression values for each gene across clusters
cluster_means_std = cluster_means.apply(lambda x: (x - x.mean()) / x.std(), axis=1)
cluster_means_std = cluster_means_std.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=1)

# Define the order of the clusters
ct_order = ['Capsular', 'Sub-Capsular', 'Cortical level I', 'Cortical level II', 'Cortical level III', 'Cortical CMJ', 'Medullar CMJ', 'Medullar level I', 'Medullar level II', 'Medullar level III', 'Medullar Deep']

# Reorder the dataframe according to the defined cluster order and transpose it
cluster_means_std = cluster_means_std[ct_order].transpose()

# Define the number of clusters and genes
num_clusters = len(cluster_means_std.index)
num_genes = len(cluster_means_std.columns)

# Set figure size
figsize = (2*num_clusters/0.5, 2*num_genes/10)  # Adjust the denominator to get the aspect ratio you want

# Perform hierarchical clustering and plot a heatmap
g = sns.clustermap(cluster_means_std, method='ward', metric='euclidean', cmap='viridis', row_cluster=False, figsize=figsize)
plt.setp(g.ax_heatmap.get_xticklabels()) 

plt.show()

# Save the plot
g.figure.savefig("bin_heatmap.pdf")

# Plot gene association with the HC 

from plotnine import ggplot, aes, geom_smooth, theme_bw, scale_color_manual
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.colors import rgb2hex
5
def plot_smoothed_line_with_normalization(adata, gene_names, obs_var, span=0.35, se=True):
    # Initialize an empty DataFrame
    data = pd.DataFrame()
    data[obs_var] = adata.obs[obs_var].values

    # Extract the gene expression for each gene, normalize it, and add it to the DataFrame
    for gene in gene_names:
        gene_expression = adata[:, gene].X.toarray().flatten()
        normalized_expression = (gene_expression -np.quantile(gene_expression,0.25)) / (np.quantile(gene_expression,0.90) - np.quantile(gene_expression,0.25))
        data[gene] = normalized_expression

    # Melt the DataFrame to long format for plotnine
    melted_data = data.melt(id_vars=[obs_var], var_name='Gene', value_name='Normalized Expression')

    # Generate a color palette with the number of colors equal to the number of genes
    color_palette = sns.color_palette('hls', len(gene_names))

    # Convert the RGB values to hexadecimal
    color_palette_hex = [rgb2hex(color) for color in color_palette]

    # Plot the smoothed lines using plotnine and add a legend
    p = (ggplot(melted_data, aes(x=obs_var, y='Normalized Expression', color='Gene')) 
         + geom_smooth(method='loess', span=span, se=se) 
         + theme_bw() 
         + scale_color_manual(values=color_palette_hex))  # Use color palette

    # Show the plot
    print(p)
    # Save the plot as a PDF
    p.save("HS_L2.pdf")

# Test the function
# gene_names = ['CHGA_mean','KERATIN_10_mean']
gene_names = ['PANCYTO_mean', 'HLADR_mean','KERATIN_10_mean','CHGA_mean','CD99_mean','KERATIN_8_mean']
gene_names = ['PANCYTO_mean','KERATIN_10_mean','KERATIN_8_mean','KERATIN_14_mean','KERATIN_15_mean']


plot_smoothed_line_with_normalization(adata_ibex_hs, gene_names, 'L2_dist_tissue_annotations_lv_1_HS')

# Guassian process enrichment 

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

def assess_enrichment_gp_series(adata, gene_name, obs_var, n_intervals=10):
    # Get the gene expression and observation values
    gene_expression = adata[:, gene_name].X.toarray().flatten()
    obs_values = adata.obs[obs_var].values

    # Initialize a Gaussian process regressor
    kernel = RBF() + WhiteKernel()
    gpr = GaussianProcessRegressor(kernel=kernel, random_state=0)

    # Fit the model
    X = obs_values.reshape(-1, 1)
    y = gene_expression
    gpr.fit(X, y)

    # Generate a series of points at which to evaluate enrichment
    min_obs = np.min(obs_values)
    max_obs = np.max(obs_values)
    intervals = np.linspace(min_obs, max_obs, n_intervals)

    # Evaluate the model at each interval
    enrichment_series = gpr.predict(intervals.reshape(-1, 1))

    return intervals, enrichment_series

gene_names = ['PANCYTO_mean', 'HLADR_mean','KERATIN_10_mean','CHGA_mean','CD99_mean','KERATIN_8_mean']

fig, ax = plt.subplots()

for gene in gene_names:
    intervals, enrichment_series = assess_enrichment_gp_series(adata_ibex_hs, gene, 'L2_dist_tissue_annotations_lv_1_HS')
    
    # Normalize the enrichment series
    normalized_enrichment = (enrichment_series - np.min(enrichment_series)) / (np.max(enrichment_series) - np.min(enrichment_series))

    ax.plot(intervals, normalized_enrichment, label=gene)

ax.set_xlabel('L2_dist_tissue_annotations_lv_1_HS')
ax.set_ylabel('Normalized Enrichment')
ax.legend()

plt.show()
