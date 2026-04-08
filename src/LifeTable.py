from SurvivalModel import SurvivalModel
import polars as pl

class LifeTable(SurvivalModel):
    def __init__(self, lx, x0 = 0, description = "Life Table"):
        self._description = description

        # Create a dataframe with columns 'x', 'l_x', 'p_x', 'q_x', 'e_x'.
        df_curated = pl.DataFrame({
            'x': x0 + pl.arange(0, len(lx)),
            'l_x': lx
        })
        
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
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(self.df_curated['x'], self.df_curated['q_x'], label='Mortality Rate q(x)')
        plt.xlabel('Age (x)')
        plt.ylabel('Mortality Rate q(x)')
        plt.title(f'Mortality Rate of {self.description}')
        plt.legend()
        plt.grid()
        plt.show()

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
        
        survival_prob = 1.0
        for year in range(t):
            current_age = x + year
            if current_age >= self.w:
                break
            q_x = self.df_curated.loc[self.df_curated['x'] == current_age, 'q_x'].values[0]
            survival_prob *= (1 - q_x)
        
        return survival_prob

    def e(self, x):
        # Return the curated remaining lifetime of (x).