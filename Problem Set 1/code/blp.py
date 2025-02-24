"""
Title: Problem Set 1 -- blp.py
Author: Benjamin Wittenbrink, Jack Kelly, Veronica Backer Peral
Date: 03/01/25
"""

import logging
import time
import numpy as np
from linearmodels.iv.model import IV2SLS
from scipy.integrate import quad_vec
from scipy.optimize import minimize
from numpy.linalg import norm

from utils import calc_nu_dist

logger = logging.getLogger(__name__)


class BLP:

    def __init__(
        self, data, tol=1e-14, max_iter=2500, numerically_integrate=False, verbose=False
    ):
        self.data = data
        self.params = data.params
        self.verbose = verbose
        self.tol = tol
        self.max_iter = max_iter
        self.numerically_integrate = numerically_integrate
        self.H = None
        self.num_moments = None

    # === Helper Methods ===
    def _integrand_probability(self, nu, sigma_alpha, delta):
        # helper function to calculate shares by integrating over nu
        num = np.exp(delta - sigma_alpha * self.data.p * nu)
        denom = 1 + np.sum(num, axis=0)
        return (num / denom) * calc_nu_dist(
            nu, self.params["nu"]["mu"], self.params["nu"]["sigma"]
        )

    def _integrand_probability_simulation(self, delta, sigma_alpha, p, nu_vec):
        delta = delta[None, :, :]
        # nu_vec = nu_vec[:, None, None]
        nu_vec = nu_vec[:, None, :]
        num = np.exp(delta - sigma_alpha * nu_vec * p)
        denom = 1 + np.sum(num, axis=1, keepdims=True)  # shape => (n_nu, 1, M)
        res = num / denom  # shape => (n_nu, J, M)
        return res

    def _invert_shares(self, sigma_alpha, nu_vec=None, tol=1e-14):
        # initialize delta to ones for contraction mapping
        delta = np.ones(self.data.jm_shape)
        true_log_shares = np.log(self.data.shares)

        if nu_vec is None and not self.numerically_integrate:
            nu_vec = np.random.lognormal(
                mean=self.params["nu"]["mu"],
                sigma=self.params["nu"]["sigma"],
                size=(self.params["nu"]["n_draws"], self.params["M"]),
            )

        for i in range(self.max_iter):
            # integrate over full nu distribution to get shares
            if self.numerically_integrate:
                shares = quad_vec(
                    lambda nu: self._integrand_probability(
                        nu, delta=delta, sigma_alpha=sigma_alpha
                    ),
                    a=0,
                    b=np.inf,
                )[0]
            else:
                shares = np.mean(
                    self._integrand_probability_simulation(
                        delta, sigma_alpha, self.data.p, nu_vec
                    ),
                    axis=0,
                )

            # update delta according to delta += true_log - predicted_log shares
            delta_new = delta + true_log_shares - np.log(shares)
            diff = np.abs(delta_new - delta).max()
            if diff < tol:
                logger.info(f"Sigma = {sigma_alpha}")
                break
            elif i % 500 == 0 and i != 0:  # print progress every 500 iterations
                diff_old = diff
                logger.info(f"Iteration {i}: max diff = {diff} (tol={tol})")
                if (diff_old - diff) < 1e-13:
                    logger.info(
                        f"Delta contraction mapping is converging slowly. Skipping. (sigma = {sigma_alpha})"
                    )
                    break
            delta = delta_new

        if i == self.max_iter - 1:
            raise Warning(
                f"Delta contraction mapping did not converge (max diff={diff})."
            )

        return delta

    def _estimate_iv_params(self, X_long, delta_long, price_long):
        # find alpha and betas
        IV_reg = IV2SLS(
            dependent=delta_long,
            exog=X_long,
            endog=price_long,
            instruments=self.H[:, 1:],
        ).fit()
        coefs = IV_reg.params.values
        betas = coefs[:3]
        alpha = -coefs[3]
        return alpha, betas

    def _estimate_xi(self, sigma_alpha, nu_vec=None):
        # Invert shares to get delta
        delta = self._invert_shares(sigma_alpha, nu_vec=nu_vec, tol=self.tol)
        delta_long = delta.flatten()
        # Estimate alpha and betas using IV regression
        alpha, betas = self._estimate_iv_params(
            self.X_long, delta_long, self.price_long
        )
        # Calculate implied xi hat
        xi = delta_long - np.dot(self.X_long, betas) + alpha * self.price_long.flatten()
        return alpha, betas, xi

    def _compute_gmm_obj(self, theta, W, nu_vec=None):
        # Compute GMM objective function
        _, _, xi = self._estimate_xi(theta[0], nu_vec=nu_vec)
        gmm_objective = (xi @ self.H).T @ W @ (xi @ self.H)
        return gmm_objective

    def _flatten(self, x):
        # If x is (C, J, M) shape, reshape to (M*J, C)
        # If x is (J,M) shape reshape to (M*J, 1)
        if len(x.shape) == 3:
            return np.reshape(x, (x.shape[0], -1)).T
        else:
            return np.reshape(x, (-1, 1))

    def _convert_data_to_long(self):
        # Flatten matrices for easy regression
        self.X_long = self._flatten(self.data.X)
        self.price_long = self._flatten(self.data.p)

    def _get_optimal_weights(self, xi):
        mat = (self.H.T * (xi.flatten() ** 2)) @ self.H
        return np.linalg.pinv(mat / (self.params["J"] * self.params["M"]))

    def construct_instruments(self):
        # form BLP instruments, which are:
        # 1. product characteristics of other products in same market
        # 2. cost shifters: W (at product level)
        # 3. cost shifters: Z (at product-market level)
        BLP_inst = np.sum(self.data.X, axis=1, keepdims=True) - self.data.X
        H = np.hstack(
            (
                self._flatten(BLP_inst),
                np.repeat(self.data.W, self.params["M"])[:, np.newaxis],
                self._flatten(self.data.Z),
            )
        )
        self.H = H
        self.num_moments = H.shape[1]

    def run_gmm_2stage(self):
        if self.verbose:
            logger.info("Estimating two-stage GMM")
            start = time.time()
        # construct instruments
        self.construct_instruments()
        # convert data to long format (X, p) for IV regression
        self._convert_data_to_long()

        if self.numerically_integrate:
            nu_vec = None
        else:
            # nu_vec = np.random.lognormal(
            #     mean=self.params["nu"]["mu"],
            #     sigma=self.params["nu"]["sigma"],
            #     size=(self.params["nu"]["n_draws"], self.params["M"]),
            # )
            nu_vec = None

        # Stage 1: Weights as identity matrix
        params_init = [1]
        weights = np.eye(self.num_moments)
        results = minimize(
            self._compute_gmm_obj,
            params_init,
            args=(
                weights,
                nu_vec,
            ),
            tol=self.tol,
            method="L-BFGS-B",
            bounds=[(1e-2, None)],
        )
        sigma_alpha = results.x[0]
        if self.verbose:
            stage1 = time.time()
            logger.info("First stage complete in %.2f seconds.", stage1 - start)

        # Stage 2: Optimal weights
        _, _, xi = self._estimate_xi(sigma_alpha, nu_vec=nu_vec)
        weights = self._get_optimal_weights(xi)
        results = minimize(
            self._compute_gmm_obj,
            [sigma_alpha],
            args=(
                weights,
                nu_vec,
            ),
            tol=self.tol,
            method="L-BFGS-B",
            bounds=[(1e-2, None)],
        )
        sigma_alpha = results.x[0]
        if self.verbose:
            stage2 = time.time()
            logger.info("Second stage complete in %.2f seconds.", stage2 - stage1)
            logger.info("Total runtime: %.2f seconds.", stage2 - start)

        alpha, beta, xi = self._estimate_xi(sigma_alpha, nu_vec=nu_vec)
        if self.verbose:
            logger.info("GMM estimation complete:")
            logger.info("\talpha_hat: %.5f", alpha)
            logger.info("\tbeta_hat: %s", np.round(beta, 5))
            logger.info("\tsigma_alpha_hat: %.5f", sigma_alpha)
        return alpha, beta, sigma_alpha
