import polars as pl

def plot_mortality_rate(*models):
    """Plot the mortality rate q(x) for multiple survival models on the same graph.

    Parameters
    ----------
    *models : SurvivalModel
        One or more instances of SurvivalModel to plot.
    """
    from plotnine import ggplot, aes, geom_line, scale_y_log10, labs, theme_minimal

    # Combine data from all models into a single DataFrame for plotting.
    mutated_dfs = [model.df_curated.with_columns(pl.lit(model.description).alias("Model")) for model in models]
    combined_df = pl.concat(mutated_dfs)

    # Create the plot using plotnine.
    plot = (ggplot(combined_df.to_pandas(), aes(x='x', y='q_x', color='Model')) +
            geom_line() +
            scale_y_log10() +
            labs(x='Age (x)', 
                 y='Mortality Rate q(x)', 
                 title='Mortality Rate Comparison of Survival Models') +
            theme_minimal())
    
    plot.show()