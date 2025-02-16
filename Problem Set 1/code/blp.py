"""
Title: Problem Set 1 -- blp.py
Author: Benjamin Wittenbrink, Jack Kelly, Veronica Backer Peral
Date: 03/01/25
"""

import numpy as np
from linearmodels.iv.model import IV2SLS
from scipy.integrate import quad_vec


class BLP:

    def __init__(self, data, params):
        self.data = data
        self.params = params

    def _compute_gmm_obj(self, theta, W):
        # helper fns: _invert_shares, _compute_moment_conditions, _construct_instruments
        delta = self._invert_shares(self, sigma_alpha, max_iter=1000, tol=1e-14)

        # Reshape X from (3,3,100) to (300 x 1)
        X_long = self.flatten(self.data.X)

        # Flatten delta to (300,)
        delta_long = delta.flatten()
        price_long = self.flatten(self.data.p)

        # Stack X_long and price_long along second dimension to create a 300 by 4 matrix
        endog = np.hstack((X_long, price_long))

        IV_reg = IV2SLS(dependent=delta_long, instruments=H, endog=endog).fit()
        betas = np.array(IV_reg.params[:3])
        alpha = IV_reg.params[3]
        xi = delta - betas @ X_long - alpha * price_long

        gmm_objective = (xi @ self.H).T @ W @ (xi @ self.H)
        return gmm_objective

    def _invert_shares(self, sigma_alpha, max_iter=1000, tol=1e-14):
        # invert shares by running delta contraction mapping
        def _calc_nu_dist(self, nu):
            return lognormal_pdf(
                nu, self.params["nu"]["mu"], self.params["nu"]["sigma"]
            )

        def integrand_probability(nu, delta, p):
            num = np.exp(delta - sigma_alpha * self.data.p * nu)
            denom = 1 + np.sum(num, axis=0)
            res = num / denom * _calc_nu_dist(nu)
            return np.exp(delta - sigma_alpha * p * nu) * _calc_nu_dist(nu)

        delta = np.ones(self.data.jm_shape)
        true_log_shares = np.log(self.data.shares)
        for i in range(max_iter):
            shares = quad_vec(
                lambda nu: integrand_probability(nu, delta=delta, p=self.data.p),
                a=0,
                b=np.inf,
            )[0]
            delta_new = delta + true_log_shares - np.log(shares)
            if np.abs(delta_new - delta).max() < tol:
                if self.verbose:
                    print(f"Delta contraction mapping converged in {i} iterations.")
                break
            delta = delta_new

        if i == max_iter - 1:
            raise Warning("Delta contraction mapping did not converge.")

        return delta

    def construct_instruments(self, **kwargs):
        # form BLP instruments
        BLP_inst = np.sum(self.data.X, axis=1, keepdims=True) - self.data.X

        H = np.hstack(
            (
                self.flatten(BLP_inst),
                self.flatten(self.data.W),
                self.flatten(self.data.Z),
            )
        )
        return H

    def flatten(self, x):

        # If x is (C, J, M) shape, reshape to (M*J, C)
        # If x is (J,M) shape reshape to (M*J, 1)

        if len(x.shape) == 3:
            return np.reshape(x, (x[0], -1)).T
        else:
            return np.reshape(x, (-1, 1))

    def run_demand_estimation(self, **kwargs):
        # simulate Pns - hold fixed for theta

        # for each theta

        # compute delta i.e., invert shares

        # solve for xi, omega

        # compute moment condition G
        pass
