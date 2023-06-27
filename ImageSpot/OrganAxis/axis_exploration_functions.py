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

