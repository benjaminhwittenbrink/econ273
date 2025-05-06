import logging
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from tqdm import tqdm
from typing import List, Dict, Tuple, Any
from linearmodels.iv.model import IV2SLS
import statsmodels.api as sm

import utils

from data import DiamondData

logger = logging.getLogger(__name__)


class DiamondModel:

    def __init__(
        self,
        data: DiamondData,
        seed: int = 123,
        verbose: bool = True,
    ):
        self.rng = np.random.default_rng(seed)

        self.DD = data
        self.data = data.to_dataframe()

        self.data["Log_H"] = np.log(self.data["High_Ed_Population"])
        self.data["Log_L"] = np.log(self.data["Low_Ed_Population"])

        self.params = self.DD.params
        self.verbose = verbose

        # Containers to be populated later
        self.theta = None
        self.W = None
        self.VCV = None
        self.g = None

        self.est_params = dict()

    # -------------------------------------------------------------------------
    def _create_instruments(self):
        """
        Create instruments for the GMM estimation.
        """
        self.data = self.data.assign(
            Z_H_reg=lambda x: x["Z_H"] * x["Regulatory_Constraint"],
            Z_H_geo=lambda x: x["Z_H"] * x["Geographic_Constraint"],
            Z_L_reg=lambda x: x["Z_L"] * x["Regulatory_Constraint"],
            Z_L_geo=lambda x: x["Z_L"] * x["Geographic_Constraint"],
        )

        self.instruments = self.data[
            ["Z_H", "Z_L", "Z_H_reg", "Z_H_geo", "Z_L_reg", "Z_L_geo"]
        ]

    # -------------------------------------------------------------------------
    #  Moment-condition builders  (TODO:)
    # -------------------------------------------------------------------------
    def _labor_demand_parameters(self) -> np.ndarray:
        """
        Run 2SLS regression to get labor demand parameters:
        Log_Wage_H = gamma_HH * log(High_Ed_Population) + gamma_HL * log(Low_Ed_Population) + epsilon_H
        Log_Wage_L = gamma_LH * log(High_Ed_Population) + gamma_LL * log(Low_Ed_Population) + epsilon_L

        instruments: Z
        """

        df = self.data

        IV_reg_H = IV2SLS(
            dependent=df["Log_Wage_H"],
            exog=None,
            endog=df[["Log_H", "Log_L"]],
            instruments=self.instruments,
        ).fit()

        self.est_params["gamma_HH"] = IV_reg_H.params.iloc[0]
        self.est_params["gamma_HL"] = IV_reg_H.params.iloc[1]

        IV_reg_L = IV2SLS(
            dependent=df["Log_Wage_L"],
            exog=None,
            endog=df[["Log_H", "Log_L"]],
            instruments=self.instruments,
        ).fit()

        self.est_params["gamma_LH"] = IV_reg_L.params.iloc[0]
        self.est_params["gamma_LL"] = IV_reg_L.params.iloc[1]

    def _housing_supply_parameters(self) -> np.ndarray:
        """
        Run 2SLS regression to get amenity supply parameters:
        Log_Rent = i+ phi*log_HD + phi_geo*log_HD*Geographic_Constraint + phi_reg*log_HD*Regulatory_Constraint + epsilon_CC
        """
        # 1. unpack parameters
        zeta = self.params["zeta"]
        df = self.data
        Z = df[["Z_H", "Z_L", "Z_H_reg", "Z_H_geo", "Z_L_reg", "Z_L_geo"]].values
        # 2. calculate residuals
        log_HD = np.log(
            zeta
            * (
                df["Low_Ed_Population"] * np.exp(df["Log_Wage_L"] - df["Log_Rent"])
                + df["High_Ed_Population"] * np.exp(df["Log_Wage_H"] - df["Log_Rent"])
            )
        )

        log_HD_x_geo = log_HD * df["Geographic_Constraint"]
        log_HD_x_reg = log_HD * df["Regulatory_Constraint"]
        endog = np.column_stack([log_HD, log_HD_x_geo, log_HD_x_reg])

        IV_reg = IV2SLS(
            dependent=df["Log_Rent"],
            exog=np.ones((len(df), 1)),
            endog=endog,
            instruments=self.instruments,
        ).fit()

        self.est_params["iota"] = np.exp(IV_reg.params.iloc[0])
        self.est_params["phi"] = IV_reg.params.iloc[0]
        self.est_params["phi_geo"] = IV_reg.params.iloc[0]
        self.est_params["phi_reg"] = IV_reg.params.iloc[0]

    def _amenity_supply_parameters(self) -> np.ndarray:
        """
        Run 2SLS regression to get amenity supply parameters:
        Amenity_Endog = phi_a * (log_H-log_L) + epsilon_
        """

        df = self.data

        IV_reg = IV2SLS(
            dependent=df["Amenity_Endog"],
            exog=None,
            endog=df["Log_H"] - df["Log_L"],
            instruments=self.instruments,
        ).fit()

        self.est_params["phi_a"] = IV_reg.params.iloc[0]

    def _labor_supply_parameters(self, delta_hat: np.ndarray) -> np.ndarray:
        """
        Run 2SLS regression to get labor supply parameters:
        delta = phi_a * (log_H-log_L) + epsilon_
        """

        df = self.data
        zeta = self.params["zeta"]

        X = [df.Log_Wage_H - zeta * df.Log_Rent, df.Log_Wage_L - zeta * df.Log_Rent]
        A = df.Amenity_Endog
        Z_full = self.instruments

        for race in self.params["race_types"]:
            # stack dependent δ’s for both edus
            ys = []
            Xs = []
            As = []
            Zs = []

            for i, edu in enumerate(self.params["edu_types"]):
                # δ is keyed by (edu, race)
                ys.append(delta_hat[(edu, race)])
                Xs.append(X[i])
                As.append(A)
                Zs.append(Z_full)

            # concatenate along the observation axis
            y_stack = np.concatenate(ys)
            X_stack = np.concatenate(Xs)
            A_stack = np.concatenate(As)
            endog = np.column_stack([X_stack, A_stack])
            Z = pd.concat(Zs, axis=0).values

            # run one IV regression per race
            iv_reg = IV2SLS(
                dependent=y_stack,
                exog=None,  # no exogenous covariates
                endog=endog,  # the two regressors whose β’s we want
                instruments=Z,
            ).fit()

            # save
            self.est_params[f"beta_w_{race}"] = iv_reg.params.iloc[0]
            self.est_params[f"beta_a_{race}"] = iv_reg.params.iloc[1]

    # -------------------------------------------------------------------------
    # BLP inner loop  (share inversion)
    # -------------------------------------------------------------------------
    def _blp_inversion(
        self, beta_st: dict, max_iter: int = 1000
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve for delta hat that reproduces observed H/L shares, given nonlinear
        preferences theta_N:
            (zeta, beta_w_White, beta_w_Black, beta_a_White, beta_a_Black, beta_st_White, beta_st_Black, phi_a).

        Return
        ------
        delta_hat : dict with mean utilities for each type
        """
        df = self.data

        # solve for delta for each type
        def objective(delta, df, edu, race):

            pop_est = self.DD._calculate_group_population(
                delta, edu, race, beta_st=beta_st
            )

            res = pop_est - df[f"Pop_{edu}{race}"]
            return (res**2).sum()

        delta_hat = {}
        for edu in self.params["edu_types"]:
            for race in self.params["race_types"]:
                delta0 = np.ones(self.params["J"])
                res = minimize(objective, delta0, args=(df, edu, race))
                delta_hat[(edu, race)] = res.x

        return delta_hat

    # -------------------------------------------------------------------------
    # GMM housekeeping TODO:
    # -------------------------------------------------------------------------
    def _stack_moments(self, theta: np.ndarray) -> np.ndarray:

        df = self.data

        beta_st = {"White": theta[0], "Black": theta[1]}
        delta_hat = self._blp_inversion(beta_st)
        zeta = self.params["zeta"]

        self._labor_demand_parameters()
        self._housing_supply_parameters()
        self._amenity_supply_parameters()
        self._labor_supply_parameters(delta_hat)

        moments = []
        for edu in self.params["edu_types"]:
            for race in self.params["race_types"]:
                d = delta_hat[(edu, race)]
                xi = (
                    d
                    - (df[f"Log_Wage_{edu}"] - zeta * df["Log_Rent"])
                    * self.est_params[f"beta_w_{race}"]
                    - df["Amenity_Endog"] * self.est_params[f"beta_a_{race}"]
                )
                g = np.mean(xi.to_numpy()[:, None] * self.instruments, axis=0)
                moments.append(g)

        return np.concatenate(moments)

    # Quadratic-form objective
    def _gmm_objective_fn(self, theta: np.ndarray, W: np.ndarray) -> float:
        g = self._stack_moments(theta)
        return g @ W @ g

    # -------------------------------------------------------------------------
    # Fit (two-step GMM)
    # -------------------------------------------------------------------------
    def initialize(self):
        self._create_instruments()

    def fit(
        self,
        theta0: np.ndarray = np.ones(2),
        outer_options: Dict[str, Any] = None,
    ):
        """
        Two-step GMM: first W = I, then optimal W = (Sigma hat)^-1 with
        residual outer-product.

        Parameters:
            gamma_HH, gamma_HL, gamma_LH, gamma_LL,
            alpha_HH, alpha_HL, alpha_LH, alpha_LL,
            zeta,
            beta_w_White, beta_w_Black,
            beta_a_White, beta_a_Black,
            beta_st_White, beta_st_Black,
            phi_a,
            phi, phi_geo, phi_reg

        """
        # -- Step 1 --
        # W = identity matrix
        if self.verbose:
            logger.info("Step 1: GMM with identity matrix.")

        W = np.eye(self._stack_moments(theta0).size)
        res1 = minimize(
            self._gmm_objective_fn,
            theta0,
            args=(W,),
            method="BFGS",
            options=outer_options or {"disp": self.verbose},
        )
        theta1 = res1.x
        g1 = self._stack_moments(theta1)

        # ── Step 2 ──  (optimal weighting)

        if self.verbose:
            logger.info(f"First stage results: {theta1.round(2)}")
            logger.info("Step 2: GMM with optimal weighting matrix.")

        Sigma_hat = np.outer(g1, g1)
        W_opt = np.linalg.inv(Sigma_hat + 1e-12 * np.eye(Sigma_hat.shape[0]))
        res2 = minimize(
            self._gmm_objective_fn,
            theta1,
            args=(W_opt,),
            method="BFGS",
            options=outer_options or {"disp": self.verbose},
        )
        theta2 = res2.x
        g2 = self._stack_moments(theta2)

        # save results
        self.theta = theta2
        self.W = W_opt
        self.g = g2

        # Compute VCV matrix ?
        logger.info("GMM estimation finished.")
