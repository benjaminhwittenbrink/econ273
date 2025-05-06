import os
import logging
import pickle
import numpy as np
import pandas as pd
import datetime as dt
from scipy.optimize import least_squares
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

    ### I/O Methods
    def write(self, file_dir=None):
        """
        Write the class and data to a folder.
        Folders will be of the form ../data/DiamondData_<seed>_<datetime>
        """
        if file_dir is None:
            file_dir = "/DiamondData"
        seed_str = "_" + str(self.seed)
        date_str = "_" + dt.datetime.now().strftime("%Y_%m_%d_%s")
        dir = "../data" + file_dir + seed_str + date_str + "/"
        if not os.path.isdir(dir):
            os.makedirs(dir)
        with open(dir + "DiamondData.pkl", "wb") as f:
            pickle.dump(self, f)
        df = self.to_dataframe()
        df.to_csv(dir + "DiamondData.csv", index=False)
        logger.info(f"Saved {self.__class__.__name__} to {dir}")

    def to_dataframe(self):
        """
        Convert the simulation results to a pandas DataFrame.
        """
        data = {
            "Population": self.population,
            "Wage_H": self.wage_H,
            "Wage_L": self.wage_L,
            "Rent": self.rent,
            "Amenity_Endog": self.amenity_endog,
            "High_Ed_Population": self.H,
            "Low_Ed_Population": self.L,
            "Regulatory_Constraint": self.x_reg,
            "Geographic_Constraint": self.x_geo,
            "Z_H": self.Z_H,
            "Z_L": self.Z_L,
        }
        return pd.DataFrame(data)

    ### Public Methods
    def simulate(self):
        logger.info("Starting simulation...")
        self._simulate_exog()
        self._simulate_endog()
        logger.info("Simulation completed.")

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

        ###### Demand Instrument Variables ######
        self.Z_H = np.random.lognormal(
            self.params["Z_H"]["mu"], self.params["Z_H"]["sigma"], self.params["J"]
        )
        self.Z_L = np.random.lognormal(
            self.params["Z_L"]["mu"], self.params["Z_L"]["sigma"], self.params["J"]
        )

        ###### Population Demographics ######
        self.high_ed = Demographic(self.params, edu_level="H")
        self.low_ed = Demographic(self.params, edu_level="L")

        ###### Amenities ######
        self.amenity_endog = np.random.lognormal(
            self.params["a"]["mu"], self.params["a"]["sigma"], self.params["J"]
        )

        self.amenity_exog = np.random.lognormal(
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

        # initialize guess for wages, rents, and amenities
        init = np.concatenate([np.ones(self.params["J"])] * 4)
        sol = least_squares(
            self._solve_prices,
            init,
            x_scale="jac",
            bounds=(1e-8, np.inf),
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
        )

        if not sol.success:
            logger.warning("Fixed point did not converge.")

        wage_L_eq, wage_H_eq, rent_eq, amenity_endog_eq = np.split(sol.x, 4)
        L_eq, H_eq, _, _, _, _ = self._find_equilibrium(
            wage_L_eq, wage_H_eq, rent_eq, amenity_endog_eq
        )
        self.L, self.H = L_eq, H_eq
        self.population = L_eq + H_eq
        self.wage_L, self.wage_H = wage_L_eq, wage_H_eq
        self.rent = rent_eq
        self.amenity_endog = amenity_endog_eq

    def _solve_prices(self, init):
        """
        Get equilibrium prices and shares.
        """
        wage_L, wage_H, rent, amenity_endog = np.split(init, 4)
        _, _, wage_L_new, wage_H_new, rent_new, amenity_endog_new = (
            self._find_equilibrium(wage_L, wage_H, rent, amenity_endog)
        )
        return np.concatenate(
            [
                wage_L_new - wage_L,
                wage_H_new - wage_H,
                rent_new - rent,
                amenity_endog_new - amenity_endog,
            ]
        )

    def _find_equilibrium(self, wage_L, wage_H, rent, amenity_endog):
        """
        For a given set of prices/amenities, find the equilibrium population.
        Then, find the prices that those populations would imply.
        """
        # Get probability of being in each city for each type
        H = self._calculate_population(self.high_ed, wage_H, rent, amenity_endog)
        L = self._calculate_population(self.low_ed, wage_L, rent, amenity_endog)

        # Update prices given population
        wage_H = (
            self.params["gamma_HH"] * np.log(H)
            + self.params["gamma_HL"] * np.log(L)
            + self.params["alpha_HH"] * np.log(self.Z_H)
            + self.params["alpha_HL"] * np.log(self.Z_L)
            + self.epsilon_H
        )

        wage_L = (
            self.params["gamma_LH"] * np.log(H)
            + self.params["gamma_LL"] * np.log(L)
            + self.params["alpha_LH"] * np.log(self.Z_H)
            + self.params["alpha_LL"] * np.log(self.Z_L)
            + self.epsilon_L
        )

        rent = self._solve_rents(H, L, wage_H, wage_L)

        # Update amenities given population
        amenity_endog = self.params["phi_a"] * np.log(H / L) + self.epsilon_a

        return L, H, wage_L, wage_H, rent, amenity_endog

    def _calculate_population(self, demographic, wage, rent, amenity_endog):
        """
        Calculate population for each city given demographic.
        """
        delta = self._get_delta(wage, rent, amenity_endog)

        util = np.exp(
            delta[demographic.race]
            + demographic.same_state
            * self.params["beta_st"][demographic.race][:, np.newaxis]
        )
        tot_util = util.sum(axis=1)
        prob = util / tot_util[:, np.newaxis]

        return prob.sum(axis=0)

    def _get_delta(self, wage, rent, amenity_endog):
        """
        Calculate delta (average utility) for each city given wage and rent.
        """
        delta = []
        for race in [0, 1]:
            d_z = (
                (wage - self.params["zeta"] * rent) * self.params["beta_w"][race]
                + amenity_endog * self.params["beta_a"][race]
                + self.amenity_exog * self.params["beta_x"][race]
            )
            delta.append(d_z)
        return np.array(delta)

    def _solve_rents(self, H, L, wage_H, wage_L, tol=1e-7):
        """
        Find the fixed point for rent given the population and wages.
        """
        J = self.params["J"]
        x0 = np.concatenate([np.ones(J), np.ones(J)])

        def rent_residuals(x):
            r, HD = x[:J], x[J:]
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

        sol = least_squares(
            rent_residuals,
            x0,
            bounds=(1e-8, np.inf),
            x_scale="jac",
            xtol=tol,
            ftol=tol,
            gtol=tol,
        )
        if not sol.success:
            logger.warning("Rent fixed point did not converge.")

        rent = sol.x[:J]
        return rent

    def _convergence_check(self, x, x_new, tol):
        """
        Check for convergence.
        """
        return np.max(np.abs(x_new - x)) < tol


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

    def to_dataframe(self):
        """
        Convert the demographic data to a pandas DataFrame.
        """
        # @TODO: implement this
        pass
