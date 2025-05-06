import logging
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from tqdm import tqdm
from typing import List, Dict, Tuple, Any

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

        self.params = self.DD.params
        self.verbose = verbose

        # Containers to be populated later
        self.theta = None
        self.theta_linear = None
        self.theta_nonlinear = None
        self.W = None
        self.VCV = None
        self.g = None

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

    # -------------------------------------------------------------------------
    #  Moment-condition builders  (TODO:)
    # -------------------------------------------------------------------------
    def _moments_labor_demand(self, theta_L: np.ndarray) -> np.ndarray:
        """
        theta_L:
            gamma_HH, gamma_HL, gamma_LH, gamma_LL,
            alpha_HH, alpha_HL, alpha_LH, alpha_LL

        Returns a (N_obs x N_instr)  moment matrix, flattened to 1-D.
        """
        # 1. unpack parameters
        (
            gamma_HH,
            gamma_HL,
            gamma_LH,
            gamma_LL,
            alpha_HH,
            alpha_HL,
            alpha_LH,
            alpha_LL,
        ) = theta_L
        df = self.data
        Z = df[
            ["Z_H", "Z_L", "Z_H_reg", "Z_H_geo", "Z_L_reg", "Z_L_geo"]
        ].values  # shape (N, 6)
        # 2. calculate residuals
        res_H = (
            df["Log_Wage_H"]
            - gamma_HH * np.log(df["High_Ed_Population"])
            - gamma_HL * np.log(df["Low_Ed_Population"])
            - alpha_HH * np.log(df["Z_H"])
            - alpha_HL * np.log(df["Z_L"])
        )
        res_L = (
            df["Log_Wage_L"]
            - gamma_LH * np.log(df["High_Ed_Population"])
            - gamma_LL * np.log(df["Low_Ed_Population"])
            - alpha_LH * np.log(df["Z_H"])
            - alpha_LL * np.log(df["Z_L"])
        )
        # 3. calculate moments and aggregate
        g_H = np.mean(res_H.to_numpy()[:, None] * Z, axis=0)
        g_L = np.mean(res_L.to_numpy()[:, None] * Z, 0)
        return np.concatenate([g_H, g_L])  # (12,)

    def _moments_housing_supply(self, theta_N: np.ndarray) -> np.ndarray:
        """Moment conditions for rent equation."""
        # 1. unpack parameters
        zeta = theta_N[0]
        phi, phi_geo, phi_reg = theta_N[8:11]
        df = self.data
        Z = df[["Z_H", "Z_L", "Z_H_reg", "Z_H_geo", "Z_L_reg", "Z_L_geo"]].values
        # 2. calculate residuals
        HD = zeta * (
            df["Low_Ed_Population"] * np.exp(df["Log_Wage_L"] - df["Log_Rent"])
            + df["High_Ed_Population"] * np.exp(df["Log_Wage_H"] - df["Log_Rent"])
        )
        phi_all = (
            phi
            + phi_geo * df["Geographic_Constraint"]
            + phi_reg * df["Regulatory_Constraint"]
        )
        res = df["Log_Rent"] - phi_all * np.log(HD)
        # 3. calculate moments and aggregate
        g = np.mean(res.to_numpy()[:, None] * Z, axis=0)
        return g

    def _moments_amenity_supply(self, theta_N: np.ndarray) -> np.ndarray:
        """Moment conditions for amenity-supply equation."""
        # 1. unpack parameters
        phi_a = theta_N[7]
        df = self.data
        Z = df[["Z_H", "Z_L", "Z_H_reg", "Z_H_geo", "Z_L_reg", "Z_L_geo"]].values
        # 2. calculate residuals
        res = df["Amenity_Endog"] - phi_a * np.log(
            df["High_Ed_Population"] / df["Low_Ed_Population"]
        )
        # 3. calculate moments and aggregate
        g = np.mean(res.to_numpy()[:, None] * Z, axis=0)
        return g

    def _moments_labor_supply(
        self, theta_N: np.ndarray, delta_hat: np.ndarray
    ) -> np.ndarray:
        """
        Uses delta_hat (mean utilities) from BLP inversion plus nonlinear paramters
        theta_N: (zeta, beta_w_White, beta_w_Black, beta_a_White, beta_a_Black, beta_st_White, beta_st_Black, phi_a)
        """
        # 1. unpack parameters
        zeta = theta_N[0]
        beta_w = {"White": theta_N[1], "Black": theta_N[2]}
        beta_a = {"White": theta_N[3], "Black": theta_N[4]}
        beta_st = {"White": theta_N[5], "Black": theta_N[6]}
        phi_a = theta_N[7]

        df = self.data
        Z = df[["Z_H", "Z_L", "Z_H_reg", "Z_H_geo", "Z_L_reg", "Z_L_geo"]].values
        # 2. calculate residuals
        g = []
        for edu in self.params["edu_types"]:
            for race in self.params["race_types"]:
                res = (
                    delta_hat[(edu, race)]
                    - (df[f"Log_Wage_H"] - zeta * df["Log_Rent"]) * beta_w[race]
                    - df["Amenity_Endog"] * beta_a[race]
                )
                g.append(np.mean(res.to_numpy()[:, None] * Z, axis=0))
        return np.concatenate(g)

    # -------------------------------------------------------------------------
    # BLP inner loop  (share inversion)
    # -------------------------------------------------------------------------
    def _blp_inversion(
        self, theta_N: np.ndarray, max_iter: int = 1000
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve for delta hat that reproduces observed H/L shares, given nonlinear
        preferences theta_N:
            (zeta, beta_w_White, beta_w_Black, beta_a_White, beta_a_Black, beta_st_White, beta_st_Black, phi_a).

        Return
        ------
        delta_hat : dict with mean utilities for each type
        """
        # 1. unpack parameters
        beta_st = {"White": theta_N[5], "Black": theta_N[6]}
        df = self.data

        # 2.  solve for delta for each type
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
    def _stack_moments(self, theta_L: np.ndarray, theta_N: np.ndarray) -> np.ndarray:
        delta_hat = self._blp_inversion(theta_N)

        g1 = self._moments_labor_demand(theta_L)
        g2 = self._moments_housing_supply(theta_N)
        g3 = self._moments_amenity_supply(theta_N)
        g4 = self._moments_labor_supply(theta_N, delta_hat)

        return np.concatenate([g1, g2, g3, g4])

    # Quadratic-form objective
    def _gmm_objective_fn(self, theta: np.ndarray, W: np.ndarray, K_L: int) -> float:
        theta_L, theta_N = theta[:K_L], theta[K_L:]
        g = self._stack_moments(theta_L, theta_N)

        # if self.verbose:
        #     logger.info(f"gamma_HH: {theta_L[0]:.3f}")
        #     logger.info(f"gamma_HL: {theta_L[1]:.3f}")
        #     logger.info(f"gamma_LH: {theta_L[2]:.3f}")
        #     logger.info(f"gamma_LL: {theta_L[3]:.3f}")
        #     logger.info(f"alpha_HH: {theta_L[4]:.3f}")
        #     logger.info(f"alpha_HL: {theta_L[5]:.3f}")
        #     logger.info(f"alpha_LH: {theta_L[6]:.3f}")
        #     logger.info(f"alpha_LL: {theta_L[7]:.3f}")
        #     logger.info(f"zeta: {theta_N[0]:.3f}")
        #     logger.info(f"beta_w: ({theta_N[1]:.3f}, {theta_N[2]:.3f})")
        #     logger.info(f"beta_a: ({theta_N[3]:.3f}, {theta_N[4]:.3f})")
        #     logger.info(f"beta_st: ({theta_N[5]:.3f}, {theta_N[6]:.3f})")
        #     logger.info(f"phi_a: {theta_N[7]:.3f}")
        #     logger.info(f"phi: {theta_N[8]:.3f}")
        #     logger.info(f"phi_geo: {theta_N[9]:.3f}")
        #     logger.info(f"phi_reg: {theta_N[10]:.3f}")
        #     logger.info(f"Loss: {np.dot(g, g):.3f}\n\n\n")

        return g @ W @ g

    # -------------------------------------------------------------------------
    # Fit (two-step GMM)
    # -------------------------------------------------------------------------
    def initialize(self):
        self._create_instruments()

    def fit(
        self,
        theta_L0: np.ndarray = np.ones(8),
        theta_N0: np.ndarray = np.ones(11),
        outer_options: Dict[str, Any] = None,
    ):
        """
        Two-step GMM: first W = I, then optimal W = (Sigma hat)^-1 with
        residual outer-product.

        Linear paramters theta_L0:
            gamma_HH, gamma_HL, gamma_LH, gamma_LL,
            alpha_HH, alpha_HL, alpha_LH, alpha_LL

        Nonlinear paramters theta_N0:
            zeta,
            beta_w_White, beta_w_Black,
            beta_a_White, beta_a_Black,
            beta_st_White, beta_st_Black,
            phi_a,
            phi, phi_geo, phi_reg

        """

        K_L = len(theta_L0)
        theta0 = np.concatenate([theta_L0, theta_N0])

        # -- Step 1 --
        # W = identity matrix
        if self.verbose:
            logger.info("Step 1: GMM with identity matrix.")

        W = np.eye(self._stack_moments(theta_L0, theta_N0).size)
        res1 = minimize(
            self._gmm_objective_fn,
            theta0,
            args=(W, K_L),
            method="BFGS",
            options=outer_options or {"disp": self.verbose},
        )
        theta1 = res1.x
        g1 = self._stack_moments(theta1[:K_L], theta1[K_L:])

        # ── Step 2 ──  (optimal weighting)

        if self.verbose:
            logger.info(f"First stage results: {theta1.round(2)}")
            logger.info("Step 2: GMM with optimal weighting matrix.")
            logger.info("Calculating outer product of residuals.")

        Sigma_hat = np.outer(g1, g1)
        W_opt = np.linalg.inv(Sigma_hat + 1e-12 * np.eye(Sigma_hat.shape[0]))
        res2 = minimize(
            self._gmm_objective_fn,
            theta1,
            args=(W_opt, K_L),
            method="BFGS",
            options=outer_options or {"disp": self.verbose},
        )
        theta2 = res2.x
        g2 = self._stack_moments(theta2[:K_L], theta2[K_L:])

        # save results
        self.theta = theta2
        self.theta_linear = theta2[:K_L]
        self.theta_nonlinear = theta2[K_L:]
        self.W = W_opt
        self.g = g2

        # Compute VCV matrix ?
        logger.info("GMM estimation finished.")
