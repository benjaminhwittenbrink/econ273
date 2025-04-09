import numpy as np
import pandas as pd
import logging
from scipy.optimize import fsolve
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DiamondData:

    def __init__(self, params, seed=123, verbose=True):
        self.seed = seed
        np.random.seed(seed)

        self.params = params
        self.verbose = verbose

        # Convert lists to arrays
        for key in params:
            if isinstance(params[key], list):
                params[key] = np.array(params[key])

    ### Subclass for each demographic group
    class Demographic:
        def __init__(self, params, edu_level="H"):
            self.params = params
            self.edu_level = edu_level
            self._initialize_params()

        def _initialize_params(self):
            # Number of individuals
            self.N = self.params[self.edu_level]["N"]

            # Set race based on probability of being black
            self.race = np.random.binomial(
                1, self.params[self.edu_level]["p_black"], self.N
            )

            # Set probability of being from that state
            self.same_state = np.random.binomial(
                1, self.params[self.edu_level]["p_state"], (self.N, self.params["J"])
            )

    ### Public Methods
    def simulate(self):
        self._simulate_exog()
        self._simulate_endog()

    ### Private Methods
    def _simulate_exog(self):
        """
        Simulate exogenous variables: housing supply, population demographics, amenities, and shocks.
        """

        ###### Housing Supply ######
        # Initialize housing supply shifters
        self.x_reg = np.random.lognormal(
            self.params["x_reg"]["mu"], self.params["x_reg"]["sigma"], self.params["J"]
        )
        self.x_geo = np.random.lognormal(
            self.params["x_geo"]["mu"], self.params["x_geo"]["sigma"], self.params["J"]
        )

        # Construction costs (time-varying)
        self.construction_costs = np.random.lognormal(
            self.params["CC"]["mu"], self.params["CC"]["sigma"], self.params["J"]
        )

        # Supply elasticity
        self.phi = (
            self.params["phi"]
            + self.params["phi_geo"] * self.x_geo
            + self.params["phi_reg"] * self.x_reg
        )

        ###### Population Demographics ######
        self.high_ed = self.Demographic(self.params, edu_level="H")
        self.low_ed = self.Demographic(self.params, edu_level="L")

        ###### Amenities ######
        self.endog_amenitiy = np.random.lognormal(
            self.params["a"]["mu"], self.params["a"]["sigma"], self.params["J"]
        )

        self.exog_amenitiy = np.random.lognormal(
            self.params["x"]["mu"], self.params["x"]["sigma"], self.params["J"]
        )

        ###### Shocks ######
        self.epsilon_H = np.random.normal(
            self.params["epsilon_H"]["mu"],
            self.params["epsilon_H"]["sigma"],
            self.params["J"],
        )
        self.epsilon_L = np.random.normal(
            self.params["epsilon_L"]["mu"],
            self.params["epsilon_L"]["sigma"],
            self.params["J"],
        )
        self.epsilon_a = np.random.normal(
            self.params["epsilon_a"]["mu"],
            self.params["epsilon_a"]["sigma"],
            self.params["J"],
        )

    def _simulate_endog(self):
        """
        Simulate endogenous variables: wages, labor supplies, prices, amenity levels.
        Where j is the city and t is the time period.
        """

        # Iterate to find fixed point
        init = np.ones(self.params["J"])
        self._run_price_fixed_point(init)

    def _run_price_fixed_point(self, init, tol=1e-10, max_iter=10000):
        """
        Get equilibrium prices and shares.
        """
        wage_L, wage_H, rent, amenity_endog = init, init, init, init

        for i in range(max_iter):
            L, H, wage_L_new, wage_H_new, rent_new, amenity_endog_new = (
                self._find_equilibrium(wage_L, wage_H, rent, amenity_endog)
            )
            # If everything has converged, break
            if (
                self._convergence_check(wage_H, wage_H_new, tol)
                and self._convergence_check(wage_L, wage_L_new, tol)
                and self._convergence_check(rent, rent_new, tol)
                and self._convergence_check(amenity_endog, amenity_endog_new, tol)
            ):
                if self.verbose:
                    logger.info(f"Fixed point converged in {i} iterations.")
                break

            # Update
            wage_H, wage_L, rent, amenity_endog = (
                wage_H_new,
                wage_L_new,
                rent_new,
                amenity_endog_new,
            )

        if i == max_iter - 1:
            print("Price fixed point did not converge.")

        self.population = H + L
        self.H = H
        self.L = L
        self.wage_H = wage_H
        self.wage_L = wage_L
        self.rent = rent
        self.amenity_endog = amenity_endog

    def _find_equilibrium(self, wage_L, wage_H, rent, amenity_endog):
        """
        For a given set of prices/amenities, find the equilibrium population.
        Then, find the prices that those populations would imply.
        """
        # Get probability of being in each city for each type
        H = self._calculate_population(self.high_ed, wage_H, rent)
        L = self._calculate_population(self.low_ed, wage_L, rent)

        # Update prices given population
        wage_H = (
            self.params["gamma_HH"] * np.log(H)
            + self.params["gamma_HL"] * np.log(L)
            + self.epsilon_H
        )

        wage_L = (
            self.params["gamma_LH"] * np.log(H)
            + self.params["gamma_LL"] * np.log(L)
            + self.epsilon_L
        )

        rent = self._rent_fixed_point(H, L, wage_H, wage_L)

        # Update amenities given population
        amenity_endog = self.params["phi_a"] * np.log(H / L) + self.epsilon_a

        return L, H, wage_L, wage_H, rent, amenity_endog

    def _calculate_population(self, demographic, wage, rent):
        """
        Calculate population for each city given demographic.
        """
        delta = self._get_delta(wage, rent)

        util = np.exp(
            delta[demographic.race]
            + demographic.same_state
            * self.params["beta_st"][demographic.race][:, np.newaxis]
        )
        tot_util = util.sum(axis=1)
        prob = util / tot_util[:, np.newaxis]

        return prob.sum(axis=0)

    def _get_delta(self, wage, rent):
        """
        Calculate delta (average utility) for each city given wage and rent.
        """
        delta = []
        for race in [0, 1]:
            d_z = (
                (wage - self.params["zeta"] * rent) * self.params["beta_w"][race]
                + self.endog_amenitiy * self.params["beta_a"][race]
                + self.exog_amenitiy * self.params["beta_x"][race]
            )
            delta.append(d_z)
        return np.array(delta)

    def _rent_fixed_point(self, H, L, wage_H, wage_L, tol=1e-7, max_iter=1000):
        """
        Find the fixed point for rent given the population and wages.
        """

        def equations(x):
            r = x[: self.params["J"]]
            HD = x[self.params["J"] :]
            eq1 = r - (
                np.log(self.params["iota"])
                + np.log(self.construction_costs)
                + self.phi * np.log(HD)
            )
            eq2 = HD - (
                L * self.params["zeta"] * np.exp(wage_L - r)
                + H * self.params["zeta"] * np.exp(wage_H - r)
            )
            return np.concatenate([eq1, eq2])

        # Solve the system using fsolve
        initial_guess = np.concatenate(
            [np.ones(self.params["J"]), np.ones(self.params["J"])]
        )
        solution = fsolve(equations, initial_guess)

        rent = solution[: self.params["J"]]
        HD = solution[self.params["J"] :]
        return rent

    def _convergence_check(self, x, x_new, tol):
        """
        Check for convergence.
        """
        return np.max(np.abs(x_new - x)) < tol
