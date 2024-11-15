from statsmodels.stats.multitest import multipletests
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

def compute_gene_statistics(adata, gene, annotation, cutoff, quantile, scale=None):
    """
    Compute gene expression statistics for a given gene.
    """
    
    from scipy.stats import zscore
    
    gene_expression = adata[:, gene].X.toarray().squeeze()
    if scale:
        gene_expression = zscore(gene_expression, axis=1)
    filtered_expression = gene_expression[gene_expression > cutoff]
    threshold = np.quantile(filtered_expression, quantile)
    cells_above_threshold = gene_expression > threshold
    total_above_threshold = cells_above_threshold.sum()
    return total_above_threshold, cells_above_threshold

def compute_p_values(adata, gene, annotation, unique_groups, total_cells, cells_above_threshold, total_above_threshold):
    """
    Compute the p-values for each group using chi-squared tests.
    """
    p_values = []
    for group in unique_groups:
        group_indices = (adata.obs[annotation] == group)
        total_group_cells = group_indices.sum()
        total_other_cells = total_cells - total_group_cells
        target_cells_above_threshold = (group_indices & cells_above_threshold).sum()
        other_cells_above_threshold = total_above_threshold - target_cells_above_threshold

        observed = np.array([[target_cells_above_threshold, other_cells_above_threshold],
                             [total_group_cells - target_cells_above_threshold, 
                              total_other_cells - other_cells_above_threshold]])

        chi2, p, _, _ = chi2_contingency(observed)
        p_values.append(p)

    return p_values

def analyze_gene_expression(adata, genes, quantile, annotation, cutoff, alpha=1e-100,scale=None):
    """
    Analyze the gene expression data, computing significant values for each gene in a list.

    Parameters:
    - adata: Annotated data matrix
    - genes: List of genes to analyze
    - quantile: Quantile threshold for filtering gene expression
    - annotation: Column name for cell type annotation
    - cutoff: Cutoff value for filtering gene expression
    - alpha: Significance level for statistical tests
    """
    unique_groups = adata.obs[annotation].unique()
    result_dict = {group: [] for group in unique_groups}
    total_cells = len(adata.obs[annotation])

    for gene in genes:
        print(f"{gene}")
        if gene not in adata.var_names:
            print(f"{gene} not found in data.")
            for group in unique_groups:
                result_dict[group].append(False)
            continue

        total_above_threshold, cells_above_threshold = compute_gene_statistics(adata, gene, annotation, cutoff, quantile,scale=None)
        p_values = compute_p_values(adata, gene, annotation, unique_groups, total_cells, cells_above_threshold, total_above_threshold)
        
        corrected_p_values = multipletests(p_values, method='bonferroni')[1]
        significant = corrected_p_values < alpha
        for group, is_significant in zip(unique_groups, significant):
            result_dict[group].append(is_significant)
    
    result_df = pd.DataFrame(result_dict).T
    result_df.columns = genes
    return result_df

def plot_heatmap(unique_df,fig_size=3,cells=None,savepath=None):
    """
    Plot a heatmap for the given DataFrame.
    """
    df_sub = unique_df.copy()
    # subset to cells 
    if cells:
        df_sub = unique_df.loc[cells,:]
    #plot 
    row,col = np.shape(df_sub)
    plt.figure(figsize=(2*fig_size*col/row, fig_size))
    sns.heatmap(df_sub, annot=True, cmap='viridis_r', linewidths=.5)
    plt.tight_layout()  # Adjust layout
    # saving 
    if savepath:
        plt.savefig(savepath)
    plt.show()

def rare_cell_marker_detection(adata, genes, quantile, annotation, cutoff, alpha=1e-100, plot=False,return_relevant_cells=True,scale=None):
    """
    Identify rare cell markers and optionally plot a heatmap.

    Parameters:
    - adata: Annotated data matrix
    - genes: List of genes to analyze
    - quantile: Quantile threshold for filtering gene expression
    - annotation: Column name for cell type annotation
    - cutoff: Cutoff value for filtering gene expression
    - alpha: Significance level for statistical tests
    - plot: Whether to plot a heatmap (default is False)
    - return_relevant_cells: returns only the cells that marker genes were found for, otherwise return all cells (default is True)
    """
    result_df = analyze_gene_expression(adata, genes, quantile, annotation, cutoff, alpha,scale=None)
    
    true_counts = result_df.sum()
    unique_df = result_df.apply(lambda col: col.map(lambda x: true_counts[col.name] if x else np.nan))
    if return_relevant_cells:
        unique_df = unique_df.dropna(how='all')

    if plot:
        plot_heatmap(unique_df)

    return unique_df


def get_markers_list(unique_df, cells, max_groups=1):
    """
    Identify the marker genes that define a specific group of cells
    This is done based on the unique_df dataframe based values <= max_groups.
    for example to get genes that are a marker for a specific cell type use - max_groups=1, for marker genes that label 2 groups - max_groups=2 etc
    Parameters:
    - unique_df: Pandas DF
    - cells: List of cells to analyze
    - max_groups: integer, explained above 
    """
    # Check if any cells are not present in the DataFrame's index
    missing_cells = [cell for cell in cells if cell not in unique_df.index]
    if missing_cells:
        print(f"Warning: Cells {missing_cells} not found in DataFrame. Ignoring these cells.")
        cells = [cell for cell in cells if cell in unique_df.index]

    marker_ind = np.where(unique_df.loc[cells] <= max_groups)[1]
    marker_genes = unique_df.columns[marker_ind]
    return np.unique(marker_genes)

    
    
def get_markers_dict(unique_df, cells, max_groups=1):
    """
    Identify the marker genes that define specific groups of cells.
    This is done based on the unique_df dataframe based values <= max_groups.
    For example to get genes that are a marker for a specific cell type use - max_groups=1, for marker genes that label 2 groups - max_groups=2 etc.
    Parameters:
    - unique_df: Pandas DF
    - cells: List of cells to analyze
    - max_groups: integer, explained above 
    """
    marker_genes_dict = {}
    for cell in cells:
        if cell not in unique_df.index:
            print(f"Warning: Cell {cell} not found in DataFrame.")
            continue
        marker_ind = np.where(unique_df.loc[cell] <= max_groups)[0]
        marker_genes = list(unique_df.columns[marker_ind])

        # Check if marker_genes is not empty before adding to the dictionary
        if marker_genes:
            marker_genes_dict[cell] = marker_genes

    return marker_genes_dict
