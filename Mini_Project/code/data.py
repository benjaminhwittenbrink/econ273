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
        date_str = "_" + dt.datetime.now().strftime("%Y_%m_%d")
        out_dir = "../data" + file_dir + seed_str + date_str + "/"
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        with open(out_dir + "DiamondData.pkl", "wb") as f:
            pickle.dump(self, f)
        city_df = self.to_dataframe()
        city_df.to_csv(out_dir + "DiamondCityData.csv", index=False)
        logger.info(f"Saved {self.__class__.__name__} to {out_dir}")

    def to_dataframe(self):
        """
        Convert the simulation results to a pandas DataFrame.
        """
        city_data = {
            "City": np.arange(self.params["J"]),
            "P_Same_State": self.prob_from_state,
            "Population": self.population,
            "Log_Wage_H": self.wage_H,
            "Log_Wage_L": self.wage_L,
            "Log_Rent": self.rent,
            "Amenity_Endog": self.amenity_endog,
            "High_Ed_Population": self.H,
            "Low_Ed_Population": self.L,
            "Regulatory_Constraint": self.x_reg,
            "Geographic_Constraint": self.x_geo,
            "Z_H": self.Z_H,
            "Z_L": self.Z_L,
        }

        for edu in self.params["edu_types"]:
            for race in self.params["race_types"]:
                city_data[f"TotPop_{edu}{race}"] = self.total_population[(edu, race)]
                city_data[f"Pop_{edu}{race}"] = self.city_population[(edu, race)]

        return pd.DataFrame(city_data)

    ### Public Methods
    def simulate(self):
        self._simulate_exog()
        self._simulate_endog()

    ### Private Methods
    def _simulate_exog(self):
        """
        Simulate exogenous variables: housing supply, population demographics, amenities, and shocks.
        """

        ###### Randomly assign cities to states ######
        self.prob_from_state = np.random.gamma(2, 2, size=self.params["J"])
        self.prob_from_state = self.prob_from_state / np.sum(self.prob_from_state)

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
        self.total_population = {
            ("H", "Black"): self.params["H"]["Black"]["N"],
            ("H", "White"): self.params["H"]["White"]["N"],
            ("L", "Black"): self.params["L"]["Black"]["N"],
            ("L", "White"): self.params["L"]["White"]["N"],
        }

        # Initialize city populations
        self.city_population = {
            ("H", "Black"): np.zeros(self.params["J"]),
            ("H", "White"): np.zeros(self.params["J"]),
            ("L", "Black"): np.zeros(self.params["J"]),
            ("L", "White"): np.zeros(self.params["J"]),
        }

        ###### Amenities ######
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
        init = np.concatenate([np.ones(self.params["J"]) for _ in range(4)])
        sol = least_squares(
            self._solve_prices,
            init,
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
        )

        if not sol.success:
            logger.warning("Fixed point did not converge.")

        wage_L_eq, wage_H_eq, rent_eq, amenity_endog_eq = np.split(sol.x, 4)

        L_eq, H_eq, wage_L_eq, wage_H_eq, rent_eq, amenity_endog_eq = (
            self._find_equilibrium(wage_L_eq, wage_H_eq, rent_eq, amenity_endog_eq)
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
        self.delta = {}
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
        H = self._calculate_population(wage_H, rent, amenity_endog, edu="H")
        L = self._calculate_population(wage_L, rent, amenity_endog, edu="L")

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

    def _calculate_group_population(self, delta, edu, race, beta_st, prob):
        """
        Calculate population for each group given delta.
        """

        pop = np.zeros(self.params["J"])

        exp_d = np.exp(delta)
        exp_d_beta = np.exp(delta + beta_st)
        tot_exp_d = exp_d.sum()

        denom = exp_d_beta + (tot_exp_d - exp_d)

        # Share of individuals from this state that will choose to live there
        term1 = prob * exp_d_beta / denom

        # Share of individuals from other states that will choose to live there
        prob_over_denom = prob / denom  # p_n / D_n
        sum_prob_over_denom = prob_over_denom.sum()  # Σ_n p_n / D_n
        term2 = exp_d * (sum_prob_over_denom - prob_over_denom)

        pop = (term1 + term2) * self.total_population[(edu, race)]
        return pop

    def _calculate_population(self, wage, rent, amenity_endog, edu="H"):
        """
        Calculate population for each city given demographic.
        """

        # For each type of individual, calculate the probability of being in each city
        tot_pop = np.zeros(self.params["J"])
        for race in self.params["race_types"]:
            delta = self._get_delta(wage, rent, amenity_endog, race=race)
            delta = delta - np.mean(delta)
            self.delta[(edu, race)] = delta

            pop = self._calculate_group_population(
                delta,
                edu,
                race,
                beta_st=self.params["beta_st"],
                prob=self.prob_from_state,
            )
            self.city_population[(edu, race)] = pop
            tot_pop += pop

        return tot_pop

    def _get_delta(self, wage, rent, amenity_endog, race="White"):
        """
        Calculate delta (average utility) for each city given wage and rent.
        """
        return (
            (wage - self.params["zeta"] * rent) * self.params["beta_w"][race]
            + amenity_endog * self.params["beta_a"][race]
            + self.amenity_exog * self.params["beta_x"][race]
        )

    def _solve_rents(self, H, L, wage_H, wage_L, tol=1e-7):
        """
        Find the fixed point for rent given the population and wages.
        """
        J = self.params["J"]
        x0 = np.concatenate([np.ones(J), np.ones(J)])

        def rent_residuals(x):
            r, log_HD = x[:J], x[J:]
            eq1 = r - (
                np.log(self.params["iota"])
                + np.log(self.construction_costs)
                + self.phi * log_HD
            )
            eq2 = np.exp(log_HD) - (
                L * self.params["zeta"] * np.exp(wage_L - r)
                + H * self.params["zeta"] * np.exp(wage_H - r)
            )
            return np.concatenate([eq1, eq2])

        sol = least_squares(rent_residuals, x0, xtol=tol, ftol=tol, gtol=tol)
        if not sol.success:
            logger.warning("Rent fixed point did not converge.")

        rent = sol.x[:J]
        return rent

    def _convergence_check(self, x, x_new, tol):
        """
        Check for convergence.
        """
        return np.max(np.abs(x_new - x)) < tol
