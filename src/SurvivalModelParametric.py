from src.SurvivalModel import SurvivalModel
import polars as pl
from chebpy import chebfun
from plotnine import *

class SurvivalModelParametric(SurvivalModel):
    def __init__(self, x0, w, S_0 = None, F_0 = None, mu = None, description = "Parametric Survival Model"):
        self._description = description
        self._x0 = x0
        self._w = w

        # Initialize the survival model based on the provided functions.
        if S_0 is not None:
            self._S_0 = chebfun(S_0, [x0, w])
            self._F_0 = 1 - self._S_0
            self._f_0 = self._F_0.diff()
            self._mu = self._f_0 / self._S_0
        elif F_0 is not None:
            self._F_0 = chebfun(F_0, [x0, w])
            self._S_0 = 1 - self._F_0
            self._f_0 = self._F_0.diff()
            self._mu = self._f_0 / self._S_0
        elif mu is not None:
            self._mu = chebfun(mu, [x0, w])
            self._S_0 = (-self._mu).exp()
            self._F_0 = 1 - self._S_0
            self._f_0 = self._F_0.diff()
        
        # Create a curated life table dataframe for ages from x0 to w.
        df_curated = pl.DataFrame({
            'x': range(x0, w + 1)
        }) \
            .with_columns(
                pl.col('x').map_elements(self.p).alias('p_x'),
                pl.col('x').map_elements(self.q).alias('q_x')
            )
        self._df_curated = df_curated

    @property
    def x0(self):
        """int: The minimum age covered by the survival model."""
        return self._x0

    @property
    def w(self):
        """int: The maximum age covered by the survival model."""
        return self._w
    
    @property
    def S_0(self):
        """chebfun: The survival function S_0(x) for the survival model."""
        return self._S_0

    @property
    def F_0(self):
        """chebfun: The cumulative distribution function F_0(x) = 1 - S_0(x) for the survival model."""
        return self._F_0
    
    @property
    def f_0(self):
        """chebfun: The probability density function f_0(x) for the survival model."""
        return self._f_0
    
    @property
    def mu(self):
        """chebfun: The force of mortality function mu(x) for the survival model."""
        return self._mu

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
        
        l_x = self._S_0(x)
        l_x_plus_t = self._S_0(x + t)

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