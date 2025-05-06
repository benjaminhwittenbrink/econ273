import logging
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from tqdm import tqdm
from typing import List, Dict, Tuple, Any

from data import DiamondData

logger = logging.getLogger(__name__)


class DiamondModel:

    def __init__(
        self,
        data: DiamondData,
        params: Dict[str, int],
        seed: int = 123,
        verbose: bool = True,
    ):
        self.rng = np.random.default_rng(seed)

        self.params = utils.dict_convert_lists_to_arrays(params)
        self.verbose = verbose

        self.data = data

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
            Z_H_reg=lambda x: x["Z_H"] * x["x_reg"],
            Z_H_geo=lambda x: x["Z_H"] * x["x_geo"],
            Z_L_reg=lambda x: x["Z_L"] * x["x_reg"],
            Z_L_geo=lambda x: x["Z_L"] * x["x_geo"],
        )

    # -------------------------------------------------------------------------
    # 3.  Moment-condition builders  (TODO:)
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
        ) = theta_L[:8]
        df = self.data
        Z = df[
            ["Z_H", "Z_L", "Z_H_reg", "Z_H_geo", "Z_L_reg", "Z_L_geo"]
        ].values  # shape (N, 6)
        # 2. calculate residuals
        res_H = (
            df["W_H"]
            - gamma_HH * df["H"]
            - gamma_HL * df["L"]
            - alpha_HH * df["Z_H"]
            - alpha_HL * df["Z_L"]
        )
        res_L = (
            df["W_L"]
            - gamma_LH * df["H"]
            - gamma_LL * df["L"]
            - alpha_LH * df["Z_H"]
            - alpha_LL * df["Z_L"]
        )
        # 3. calculate moments and aggregate
        g_H = np.mean(res_H[:, None] * Z, axis=0)
        g_L = np.mean(res_L[:, None] * Z, 0)
        return np.concatenate([g_H, g_L])  # (12,)

    def _moments_housing_supply(self, theta_L: np.ndarray) -> np.ndarray:
        """Moment conditions for rent equation."""
        # 1. unpack parameters
        varphi, varphi_geo, varphi_reg, zeta = theta_L[8:12]
        df = self.data
        Z = df[["Z_H", "Z_L", "Z_H_reg", "Z_H_geo", "Z_L_reg", "Z_L_geo"]].values
        # 2. calculate residuals
        # @TODO: check if this is correct (i.e., little w vs. divide by R?)
        HD = zeta * (df["L"] * df["W_L"] + df["H"] * df["W_H"])
        varphi_all = (
            varphi + varphi_geo * np.exp(df["x_geo"]) + varphi_reg * np.exp(df["x_reg"])
        )
        res = df["Rent"] - varphi_all * HD
        # 3. calculate moments and aggregate
        g = np.mean(res[:, None] * Z, axis=0)
        return g

    def _moments_amenity_supply(self, theta_L: np.ndarray) -> np.ndarray:
        """Moment conditions for amenity-supply equation."""
        # 1. unpack parameters
        varphi_a = theta_L[12]
        df = self.data
        Z = df[["Z_H", "Z_L", "Z_H_reg", "Z_H_geo", "Z_L_reg", "Z_L_geo"]].values
        # 2. calculate residuals
        res = df["Amenity_endog"] - varphi_a * np.log(df["H"] / df["L"])
        # 3. calculate moments and aggregate
        g = np.mean(res[:, None] * Z, axis=0)
        return g

    def _moments_labor_supply(
        self, theta_L: np.ndarray, delta_hat: np.ndarray
    ) -> np.ndarray:
        """
        Uses delta_hat (mean utilities) from BLP inversion plus xi and beta_st in theta_L.
        """
        # 1. unpack parameters
        zeta = theta_L[11]
        beta_w, beta_a, beta_x = theta_L[13:16]
        df = self.data
        Z = df[["Z_H", "Z_L", "Z_H_reg", "Z_H_geo", "Z_L_reg", "Z_L_geo"]].values
        # 2. calculate residuals
        # @TODO: need the worker level data here
        raise NotImplementedError

    # -------------------------------------------------------------------------
    # 4.  BLP inner loop  (share inversion) TODO:
    # -------------------------------------------------------------------------
    def _blp_inversion(
        self, theta_L: np.ndarray, theta_N: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve for delta hat that reproduces observed H/L shares, given nonlinear
        preferences theta_N:
            (beta_w, beta_a, beta_x, zeta, phi_a).

        Return
        ------
        delta_hat : (N_obs,)  mean utilities
        xi_hat : (N_obs,)  unobserved component used in moments
        """
        # TODO: contraction mapping or fixed-point
        raise NotImplementedError

    # -------------------------------------------------------------------------
    # 5.  GMM housekeeping TODO:
    # -------------------------------------------------------------------------
    def _stack_moments(self, theta_L: np.ndarray, theta_N: np.ndarray) -> np.ndarray:
        delta_hat, _ = self._blp_inversion(theta_L, theta_N)

        g1 = self._moments_labor_demand(theta_L)
        g2 = self._moments_housing_supply(theta_L)
        g3 = self._moments_amenity_supply(theta_L)
        g4 = self._moments_labor_supply(theta_L, delta_hat)

        return np.concatenate([g1, g2, g3, g4])

    # Quadratic-form objective
    def _gmm_objective_fn(self, theta: np.ndarray, W: np.ndarray, K_L: int) -> float:
        theta_L, theta_N = theta[:K_L], theta[K_L:]
        g = self._stack_moments(theta_L, theta_N)
        return g @ W @ g

    # -------------------------------------------------------------------------
    # 6.  Fit (two-step GMM)
    # -------------------------------------------------------------------------
    def fit(
        self,
        theta_L0: np.ndarray,
        theta_N0: np.ndarray,
        outer_options: Dict[str, Any] = None,
    ):
        """
        Two-step GMM: first W = I, then optimal W = (Sigma hat)^-1 with
        residual outer-product.
        """
        K_L = len(theta_L0)
        theta0 = np.concatenate([theta_L0, theta_N0])

        # -- Step 1 --
        # W = identity matrix
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
