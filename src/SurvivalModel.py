from abc import ABC, abstractmethod 

class SurvivalModel(ABC): 
    #/!\ This is an abstract class, it should not be instantiated directly.

    # Abstarct properties that should be implemented by subclasses.
    @property 
    @abstractmethod 
    def x0(self): 
        # Minimal age of the population.
        pass

    @property 
    @abstractmethod 
    def w(self): 
        # Maximal age of the population.
        pass

    @property 
    @abstractmethod 
    def description(self): 
        # A short description of the model (used in visuals).
        pass

    @property 
    @abstractmethod 
    def df_curated(self): 
        # A dataframe containing the curated data of the model, with columns 'x', 'p_x', 'q_x', 'e_x'.
        pass

    # Abstract method that should be implemented by subclasses.
    @abstractmethod 
    def plot(self): 
        # Plot the mortality rate q(x) of the model.
        pass

    @abstractmethod
    def q(self, x, t = 1, u = 0):
        # Return the probability that (x) survives u years and than dies in subsequent t years. 
        pass

    @abstractmethod
    def p(self, x, t = 1):
        # Return the probability that (x) survives t years. 
        pass

    @abstractmethod
    def e(self, x):
        # Return the curated remaining lifetime of (x).
        pass     



