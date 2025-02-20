"""
Title: Problem Set 1 -- simulate.py
Author: Benjamin Wittenbrink, Jack Kelly, Veronica Backer Peral
Date: 03/01/25
"""

import logging
import numpy as np
from scipy.integrate import quad_vec

from utils import calc_nu_dist

logger = logging.getLogger(__name__)


class DemandData:

    def __init__(self, params, seed=14_273, verbose=False):
        self.params = params
        self.seed = seed
        self.verbose = verbose
        self.jm_shape = (self.params["J"], self.params["M"])

        # initialize data attributes
        self.X = None
        self.Z = None
        self.W = None
        self.xi = None
        self.eta = None
        self.mc = None

        np.random.seed(self.seed)

    # === Helper Methods ===

    # Initialize products, cost shifters, and shocks
    def _initialize_products(self):
        # construct product characteristics
        X_dist = self.params["X"]
        X1 = np.ones(self.jm_shape)
        X2 = np.random.uniform(X_dist["X2"]["a"], X_dist["X2"]["b"], self.jm_shape)
        X3 = np.random.normal(X_dist["X3"]["mu"], X_dist["X3"]["sigma"], self.jm_shape)
        self.X = np.stack((X1, X2, X3))

    def _initialize_cost_shifters(self):
        # construct cost shifters
        cdist = self.params["cost"]
        self.Z = np.random.lognormal(
            cdist["Z"]["mu"], cdist["Z"]["sigma"], self.jm_shape
        )
        self.W = np.random.lognormal(
            cdist["W"]["mu"], cdist["W"]["sigma"], self.params["J"]
        )

    def _initialize_shocks(self):
        self.xi = np.random.normal(
            self.params["xi"]["mu"], self.params["xi"]["sigma"], self.jm_shape
        )
        self.eta = np.random.lognormal(
            self.params["cost"]["eta"]["mu"],
            self.params["cost"]["eta"]["sigma"],
            self.jm_shape,
        )

    def _set_marginal_cost(self):
        gammas = self.params["gammas"]
        self.mc = (
            gammas[0]
            + gammas[1] * self.W[:, np.newaxis]
            + gammas[2] * self.Z
            + self.eta
        )

    # Helper methods to calculate shares
    def _calc_util(self, p, nu, delta):
        return np.exp(delta - self.params["sigma_alpha"] * nu * p)

    def _integrand_probability(self, nu, delta, p):
        # probability is exp(U) / (1 + sum_(k) exp(U))
        num = self._calc_util(p, nu, delta)
        denom = 1 + np.sum(num, axis=0)
        # multiply probability with density of nu
        res = num / denom
        res *= calc_nu_dist(nu, self.params["nu"]["mu"], self.params["nu"]["sigma"])
        return res

    # Helper methods to calculate derivative of shares
    def _integrand_derivative(self, nu, delta, p):
        # derivative is (1 + sum_(k != j) exp(U)) / (1 + sum_(k) exp(U))^2 * dU_ijm/dp_jm
        util = self._calc_util(p, nu, delta)
        tot_util = 1 + np.sum(util, axis=0)
        num = tot_util - util
        denom = (tot_util) ** 2
        dU_dp = -util * (self.params["alpha"] + self.params["sigma_alpha"] * nu)
        # multiply with density of nu
        res = (num / denom) * dU_dp
        res *= calc_nu_dist(nu, self.params["nu"]["mu"], self.params["nu"]["sigma"])
        return res

    def _integrand_probability_vectorized(self, delta, p, nu):
        delta = delta[None, :, :]
        nu = nu[:, None, None]
        num = self._calc_util(p, nu, delta)  # shape => (n_nu, J, M)
        denom = 1 + np.sum(num, axis=1, keepdims=True)  # shape => (n_nu, 1, M)
        res = num / denom  # shape => (n_nu, J, M)
        return res

    def _integrand_derivative_vectorized(self, delta, p, nu):
        delta = delta[None, :, :]
        nu = nu[:, None, None]
        util = self._calc_util(p, nu, delta)
        tot_util = 1 + np.sum(util, axis=1, keepdims=True)
        num = tot_util - util
        denom = (tot_util) ** 2
        dU_dp = -util * (self.params["alpha"] + self.params["sigma_alpha"] * nu)
        # multiply with density of nu
        res = (num / denom) * dU_dp
        return res

    # === Public Methods ===
    def simulate(self, init_p=None, tol=1e-14, max_iter=1000):
        """
        Simulate data for the model.
        This function generates simulated data and returns prices and market shares.

        Returns
        -------
        shares : array_like, shape (J , M)
                Market shares.
        p : array_like, shape (J, M)
            Prices.
        """
        self._initialize_products()
        self._initialize_cost_shifters()
        self._initialize_shocks()
        self._set_marginal_cost()

        if init_p is None:
            init_p = self.mc

            # init_p = get_logit_p(data)
        self.shares, self.p, self.delta = self.run_price_fixed_point(
            init_p, tol=tol, max_iter=max_iter
        )

        return self.shares, self.p, self.delta

    def run_price_fixed_point(self, init_p, tol=1e-6, max_iter=1000, num_draws=1000):
        """
        Find the fixed point of prices using an iterative approach.
        Integrates shares, derivative of shares over distribution of nus.
        Calculates price according to oligopolistic pricing equation:
            (p - mc) / p = -1 / (d ln s / d ln p)
        which we transform to:
            p = - s / (d s / d p) + mc

        Parameters
        ----------
        init_p : array_like, shape (J, M)
            Initial prices.
        tol : float, optional
            Tolerance for convergence. Default is 1e-6.
        max_iter : int, optional
            Maximum number of iterations. Default is 1000.

        Returns
        -------
        shares : array_like, shape (J, M)
            Market shares corresponding to the converged prices.
        p : array_like, shape (J, M)
            Converged prices.
        """
        p = init_p

        nu = np.random.lognormal(
            mean=self.params["nu"]["mu"],
            sigma=self.params["nu"]["sigma"],
            size=num_draws,
        )

        for i in range(max_iter):
            shares, ds_dp, delta = self.derive_shares(p, nu_vec=nu)
            # update price according to oligopolistic pricing equation
            p_new = -shares / ds_dp + self.mc
            # if price converges, exit loop, else continue
            if np.abs(p_new - p).max() < tol:
                if self.verbose:
                    logger.info(f"Price fixed point converged in {i} iterations.")
                break
            p = p_new

        if i == max_iter - 1:
            raise Warning("Price fixed point did not converge.")

        return shares, p, delta

    def derive_shares(self, p, numerically_integrate=False, nu_vec=None):
        """
        Derive market shares and derivative of market shares wrt to prices.

        Parameters
        ----------
        p : array_like, shape (J, M)
            Prices.

        Returns
        -------
        shares : array_like, shape (J, M)
            Market shares.
        ds_dp : array_like, shape (J, M)
            Derivative of market shares wrt to prices.
        """
        betas = np.array(self.params["betas"])
        delta = np.tensordot(betas, self.X, axes=1) - self.params["alpha"] * p + self.xi

        if numerically_integrate:
            # numerically integrate over full support of nu distribution
            shares = quad_vec(
                lambda nu: self._integrand_probability(nu, delta=delta, p=p),
                a=0,
                b=np.inf,
            )[0]
            ds_dp = quad_vec(
                lambda nu: self._integrand_derivative(nu, delta=delta, p=p),
                a=0,
                b=np.inf,
            )[0]
        else:
            # simulate data to get probability
            shares = np.mean(
                self._integrand_probability_vectorized(delta, p, nu_vec),
                axis=0,
            )
            ds_dp = np.mean(
                self._integrand_derivative_vectorized(delta, p, nu_vec), axis=0
            )

        return shares, ds_dp, delta

    # === Moment Conditions ===

    # Helper methods to compute moment conditions
    @staticmethod
    def _leave_one_out_mean(x, axis, n):
        x_sum = np.sum(x, axis=axis, keepdims=True)
        return (x_sum - x) / (n - 1)

    @staticmethod
    def _calc_xi_X_moment(xi, X, axes, norm):
        return np.sum(xi * X, axis=axes) / norm

    def compute_empirical_moments(self):
        M, J = self.params["M"], self.params["J"]
        # compute E[xi * X]
        # sum across M, across J and then divide by J * M to get mean
        E_xi_X = self._calc_xi_X_moment(self.xi, self.X, axes=(1, 2), norm=M * J)

        # compute E[xi * loo_mean(X)]
        # calculate leave one out mean of each characteristic for products in same market (i.e., over J)
        loo_mean_X = self._leave_one_out_mean(self.X, axis=1, n=J)
        E_xi_loo_mean_X = self._calc_xi_X_moment(
            self.xi, loo_mean_X, axes=(1, 2), norm=M * J
        )

        # compute E[xi * p]
        E_xi_p = np.mean(self.xi * self.p)

        # compute E[xi * loo_mean(p)]
        # calculate leave one out mean of prices for products in same market (i.e., over J)
        loo_mean_p = self._leave_one_out_mean(self.p, axis=0, n=J)
        E_xi_loo_mean_p = np.mean(self.xi * loo_mean_p)

        return {
            "E_xi_X": E_xi_X,
            "E_xi_loo_mean_X": E_xi_loo_mean_X,
            "E_xi_p": E_xi_p,
            "E_xi_loo_mean_p": E_xi_loo_mean_p,
        }
