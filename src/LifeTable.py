from src.SurvivalModel import SurvivalModel
import polars as pl
from plotnine import *

class LifeTable(SurvivalModel):
    def __init__(self, lx, x0 = 0, description = "Life Table"):
        self._description = description

        # Create a dataframe with columns 'x', 'l_x', 'p_x', 'q_x', 'e_x'.
        df_curated = pl.DataFrame({
            'l_x': lx
        }) \
            .with_columns(
                pl.int_range(x0, x0 + len(lx)).alias('x'),
                pl.col('l_x').shift(-1, fill_value = 0).alias('l_x_next')
            ) \
            .with_columns(
                (pl.col('l_x_next') / pl.col('l_x')).alias('p_x')
            ) \
            .with_columns(
                (1 - pl.col('p_x')).alias('q_x')
            ) \
            .sort('x', descending = True) \
            .with_columns(
                pl.col('p_x').cum_sum().alias('e_x')
            ) \
            .sort('x') \
            .select(['x', 'l_x', 'p_x', 'q_x', 'e_x'])
        
        self._df_curated = df_curated

    @property 
    def x0(self): 
        return self._df_curated['x'].min()

    @property 
    def w(self): 
        return self._df_curated['x'].max()

    @property 
    def description(self): 
        return self._description

    @property 
    def df_curated(self): 
        return self._df_curated

    def plot(self):
        # Plot the mortality rate q(x) of the model.
        
        plot = (ggplot(self.df_curated.to_pandas(), aes(x='x', y='q_x')) +
            geom_line() +
            scale_y_log10() +
            labs(x='Age (x)', 
                 y='Mortality Rate q(x)', 
                 title=f'Mortality Rate of {self.description}') +
            theme_minimal())
        
        plot.show()

    def q(self, x, t = 1, u = 0):
        # Return the probability that (x) survives u years and than dies in subsequent t years. 
        if x < self.x0 or x > self.w:
            raise ValueError("Age x is out of bounds.")
        
        p_survive_u_years = self.p(x, u)
        p_die_in_t_years_after_u = 1 - self.p(x + u, t)
        
        return p_survive_u_years * p_die_in_t_years_after_u

    def p(self, x, t = 1):
        # Return the probability that (x) survives t years. 
        if x < self.x0 or x > self.w:
            raise ValueError("Age x is out of bounds.")
        
        if t == 0:
            return 1
        
        # Using l_x to calculate the survival probability.
        l_x = self.df_curated.filter(pl.col('x') == x).item(0, 'l_x')
        l_x_plus_t = self.df_curated.filter(pl.col('x') == x + t).item(0, 'l_x')

        return l_x_plus_t / l_x 

    def e(self, x):
        # Return the curated remaining lifetime of (x).
        if x < self.x0 or x > self.w:
            raise ValueError("Age x is out of bounds.")
        return self.df_curated.filter(pl.col('x') == x).item(0, 'e_x')