from src.SurvivalModel import SurvivalModel
import polars as pl
from plotnine import *

class SurvivalModelParametric(SurvivalModel):
    def __init__(self, x0, w, S_0 = None, F_0 = None, mu = None, description = "Parametric Survival Model"):
        self._description = description
        self._x0 = x0
        self._w = w

        

    @property
    def x0(self):
        """int: The minimum age covered by the survival model."""
        return self._x0

    @property
    def w(self):
        """int: The maximum age covered by the survival model."""
        return self._w

    @property
    def description(self):
        """str: The description provided at initialization."""
        return self._description

    @property
    def df_curated(self):
        """polars.DataFrame: The curated life table containing x, l_x, p_x, q_x, and e_x."""
        return self._df_curated

    def plot(self):
        """Render a mortality rate plot for the life table.

        The plot displays q(x) on a log scale against age x.
        """
        plot = (ggplot(self.df_curated.to_pandas(), aes(x='x', y='q_x')) +
            geom_line() +
            scale_y_log10() +
            labs(x='Age (x)', 
                 y='Mortality Rate q(x)', 
                 title=f'Mortality Rate of {self.description}') +
            theme_minimal())
        
        plot.show()

    def q(self, x, t = 1, u = 0):
        """Return the probability that an individual aged x dies after surviving u years and within the next t years.

        Parameters
        ----------
        x : int
            The starting age.
        t : int, default 1
            Number of years in the death interval following survival of u years.
        u : int, default 0
            Years survived before the death interval begins.

        Returns
        -------
        float
            Probability that (x) survives u years then dies within the following t years.
        """
        if x < self.x0 or x > self.w:
            raise ValueError("Age x is out of bounds.")
        
        p_survive_u_years = self.p(x, u)
        p_die_in_t_years_after_u = 1 - self.p(x + u, t)
        
        return p_survive_u_years * p_die_in_t_years_after_u

    def p(self, x, t = 1):
        """Return the probability that an individual aged x survives t years.

        Parameters
        ----------
        x : int
            The starting age.
        t : int, default 1
            The number of years to survive.

        Returns
        -------
        float
            Survival probability p(x, t).
        """
        if x < self.x0 or x > self.w:
            raise ValueError("Age x is out of bounds.")
        
        if t == 0:
            return 1
        
        l_x = self.df_curated.filter(pl.col('x') == x).item(0, 'l_x')
        l_x_plus_t = self.df_curated.filter(pl.col('x') == x + t).item(0, 'l_x')

        return l_x_plus_t / l_x 

    def e(self, x):
        """Return the expected remaining lifetime for an individual aged x.

        Parameters
        ----------
        x : int
            The age for which to compute remaining lifetime.

        Returns
        -------
        float
            Remaining lifetime e_x from the curated life table.
        """
        if x < self.x0 or x > self.w:
            raise ValueError("Age x is out of bounds.")
        return self.df_curated.filter(pl.col('x') == x).item(0, 'e_x')