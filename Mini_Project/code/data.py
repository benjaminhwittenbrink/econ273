import numpy as np
import pandas as pd


class DiamondData:

    def __init__(self, seed, params):
        self.seed = seed
        np.random.seed(seed)

        self.params = params

    ### Public Methods
    def simulate(self):
        pass

    ### Private Methods
    def _simulate_exog(self):

        self._simulate_wages()
        self._simulate_populations()
        self._simulate_prices()
        # self._simulate_amenities()
        # self._simulate_consumers()

    def _simulate_wages(self):
        """
        Simulate wages for high and low skill workers: w^H_{j,t} and w^L_{j,t}
        Where j is the city and t is the time period.
        """
        self.w_H_jt = None
        self.w_L_jt = None

    def _simulate_populations(self):
        """
        Simulate populations for high and low skill workers: H_{j,t} and L_{j,t}
        Where j is the city and t is the time period.
        """
        self.H_jt = None
        self.L_jt = None

    def _simulate_prices(self):
        """
        Simulate rents p_{j, t} and ...
        Where j is the city and t is the time period.
        """
        pass

    ...  # Other methods
