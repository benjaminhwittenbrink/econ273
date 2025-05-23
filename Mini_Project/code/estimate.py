import logging
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from tqdm import tqdm
from typing import List, Dict, Tuple, Any
from linearmodels.iv.model import IV2SLS
import statsmodels.api as sm
import matplotlib.pyplot as plt
import copy

import utils

from data import DiamondData

logger = logging.getLogger(__name__)


class DiamondModel:

    def __init__(
        self,
        df: pd.DataFrame,
        DD: DiamondData,
        seed: int = 123,
        verbose: bool = True,
    ):
        self.rng = np.random.default_rng(seed)

        self.DD = DD
        self.data = df

        self.params = copy.deepcopy(self.DD.params)
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
            Log_Z_H=lambda x: np.log(x["Z_H"]),
            Log_Z_L=lambda x: np.log(x["Z_L"]),
            Z_H_reg=lambda x: np.log(x["Z_H"]) * x["Regulatory_Constraint"],
            Z_H_geo=lambda x: np.log(x["Z_H"]) * x["Geographic_Constraint"],
            Z_L_reg=lambda x: np.log(x["Z_L"]) * x["Regulatory_Constraint"],
            Z_L_geo=lambda x: np.log(x["Z_L"]) * x["Geographic_Constraint"],
        )

        self.instruments_full = self.data[
            ["Log_Z_H", "Log_Z_L", "Z_H_reg", "Z_H_geo", "Z_L_reg", "Z_L_geo"]
        ]
        # self.instruments = self.data[["Log_Z_H", "Log_Z_L"]]
        self.instruments = self.instruments_full.copy()
        self.instruments_interact = self.data[
            ["Z_H_reg", "Z_H_geo", "Z_L_reg", "Z_L_geo"]
        ]
        self.instruments_sub = self.data[["Log_Z_H", "Log_Z_L"]]

    # -------------------------------------------------------------------------
    #  Moment-condition builders
    # -------------------------------------------------------------------------
    def _labor_demand_parameters(self) -> None:
        """
        Run 2SLS regression to get labor demand parameters:
        Log_Wage_H = gamma_HH * log(High_Ed_Population) + gamma_HL * log(Low_Ed_Population) + epsilon_H
        Log_Wage_L = gamma_LH * log(High_Ed_Population) + gamma_LL * log(Low_Ed_Population) + epsilon_L

        instruments: Z
        """

        df = self.data

        IV_reg_H = IV2SLS(
            dependent=df["Log_Wage_H"],
            exog=sm.add_constant(df[["Log_Z_H", "Log_Z_L"]]),
            endog=df[["Log_H", "Log_L"]],
            instruments=self.instruments_interact,
        ).fit()

        self.est_params["alpha_HH"] = IV_reg_H.params.iloc[1]
        self.est_params["alpha_HL"] = IV_reg_H.params.iloc[2]

        self.est_params["gamma_HH"] = IV_reg_H.params.iloc[3]
        self.est_params["gamma_HL"] = IV_reg_H.params.iloc[4]

        IV_reg_L = IV2SLS(
            dependent=df["Log_Wage_L"],
            exog=sm.add_constant(df[["Log_Z_H", "Log_Z_L"]]),
            endog=df[["Log_H", "Log_L"]],
            instruments=self.instruments_interact,
        ).fit()

        self.est_params["alpha_LH"] = IV_reg_L.params.iloc[1]
        self.est_params["alpha_LL"] = IV_reg_L.params.iloc[2]
        self.est_params["gamma_LH"] = IV_reg_L.params.iloc[3]
        self.est_params["gamma_LL"] = IV_reg_L.params.iloc[4]

    def _housing_supply_parameters(self, zeta) -> np.ndarray:
        """
        Run 2SLS regression to get amenity supply parameters:
        Log_Rent = i+ phi*log_HD + phi_geo*log_HD*Geographic_Constraint + phi_reg*log_HD*Regulatory_Constraint + epsilon_CC
        """
        df = self.data
        log_HD = np.log(
            zeta
            * (
                df["Low_Ed_Population"] * np.exp(df["Log_Wage_L"] - df["Log_Rent"])
                + df["High_Ed_Population"] * np.exp(df["Log_Wage_H"] - df["Log_Rent"])
            )
        )

        y = df["Log_Rent"]
        log_HD_x_geo = log_HD * df["Geographic_Constraint"]
        log_HD_x_reg = log_HD * df["Regulatory_Constraint"]
        endog = np.column_stack([log_HD, log_HD_x_geo, log_HD_x_reg])

        IV_reg = IV2SLS(
            dependent=y,
            exog=np.ones((len(df), 1)),
            endog=endog,
            instruments=self.instruments_full,
        ).fit()

        self.est_params["iota"] = np.exp(IV_reg.params.iloc[0])
        self.est_params["phi"] = IV_reg.params.iloc[1]
        self.est_params["phi_geo"] = IV_reg.params.iloc[2]
        self.est_params["phi_reg"] = IV_reg.params.iloc[3]

        # Get residuals for moment conditions
        res = sm.OLS(y, sm.add_constant(endog)).fit()
        g = np.mean(res.resid.to_numpy()[:, None] * self.instruments_full, axis=0)
        return g.to_numpy()

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

    def _estimate_2sls(self, delta_hat, df, zeta, race="White"):
        # stack high‑ and low‑education observations
        y_H = delta_hat[("H", race)]
        y_L = delta_hat[("L", race)]
        y = np.concatenate([y_H, y_L])

        # construct regressors
        X_H = df.Log_Wage_H - (zeta * df.Log_Rent)
        X_L = df.Log_Wage_L - (zeta * df.Log_Rent)
        A_H = df.Amenity_Endog
        A_L = df.Amenity_Endog

        X = pd.concat(
            [
                pd.DataFrame({"wage_diff": X_H, "amenity": A_H}),
                pd.DataFrame({"wage_diff": X_L, "amenity": A_L}),
            ]
        )

        # Add fixed effects to control for level differences by skill groups (since delta is identified up to a constant)
        FE = pd.concat(
            [
                pd.DataFrame({"FE1": np.ones(len(y_H)), "FE2": np.zeros(len(y_L))}),
                pd.DataFrame({"FE1": np.zeros(len(y_H)), "FE2": np.ones(len(y_L))}),
            ]
        )

        X = pd.concat([X, FE], axis=1)

        # instruments: just repeat the same instrument matrix for H and L
        Z = pd.concat([self.instruments, self.instruments])

        # run 2SLS:
        res_IV = IV2SLS(
            dependent=y,
            exog=X[["FE1", "FE2"]],
            endog=X[["wage_diff", "amenity"]],
            instruments=Z,
        ).fit()

        # iv_resids = res_IV.resids.to_numpy()

        # Get residuals for moment conditions
        res = sm.OLS(y, X).fit()
        g = np.mean(res.resid.to_numpy()[:, None] * Z, axis=0)

        return (res_IV.params.iloc[2], res_IV.params.iloc[3], g, res.resid.to_numpy())

    def _labor_supply_parameters(self, delta_hat: np.ndarray, zeta) -> np.ndarray:
        """
        Run 2SLS regression to get labor supply parameters:
        delta = (wage - zeta * rent) * beta_w + amenity * beta_a + epsilon
        """

        df = self.data
        moments = []

        self.est_params["beta_w"] = {}
        self.est_params["beta_a"] = {}

        for race in self.params["race_types"]:
            beta_w, beta_a, g, _ = self._estimate_2sls(delta_hat, df, zeta, race=race)
            moments.append(g)

            # save
            self.est_params["beta_w"][race] = beta_w
            self.est_params["beta_a"][race] = beta_a

        return np.concatenate(moments)

    # -------------------------------------------------------------------------
    # BLP inner loop  (share inversion)
    # -------------------------------------------------------------------------
    def _invert_delta(
        self, target_share, delta0, edu, race, beta_st, tol=1e-10, maxiter=1000
    ):
        delta = delta0.copy()
        for i in range(maxiter):
            share_pred = self.DD._calculate_group_population(
                delta, edu, race, beta_st, self.data["P_Same_State"].to_numpy()
            )
            delta_new = delta + np.log(target_share) - np.log(share_pred)
            if np.max(np.abs(delta_new - delta)) < tol:
                break
            delta = delta_new

        if i == maxiter - 1:
            logger.info(
                f"Delta contraction mapping did not converge (max diff={np.max(np.abs(delta_new - delta))})."
            )

        return np.array(delta)

    def _blp_inversion(self, beta_st: dict) -> np.ndarray:
        """
        Solve for delta hat that reproduces observed H/L shares, given nonlinear
        preferences beta_st

        Return
        ------
        delta_hat : dict with mean utilities for each type
        """
        df = self.data
        N = len(df)
        delta0 = np.ones(N)

        delta_hat = {}
        for edu in self.params["edu_types"]:
            for race in self.params["race_types"]:
                d = self._invert_delta(
                    df[f"Pop_{edu}{race}"], delta0, edu, race, beta_st
                )
                delta_hat[(edu, race)] = d

        return delta_hat

    # -------------------------------------------------------------------------
    # GMM housekeeping
    # -------------------------------------------------------------------------
    def _stack_moments(self, theta: np.ndarray) -> np.ndarray:
        beta_st, zeta = theta
        delta_hat = self._blp_inversion(beta_st)
        moments1 = self._labor_supply_parameters(delta_hat, zeta)
        moments2 = self._housing_supply_parameters(zeta)
        moments = np.concatenate([moments1, moments2])

        return moments

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
        outer_options: Dict[str, Any] = {"disp": False},
    ):
        """
        Two-step GMM: first W = I, then optimal W = (Sigma hat)^-1 with
        residual outer-product.
        """

        method = "Nelder-Mead"  # "L-BFGS-B"  # "Powell" #

        # -- Step 1 --
        # W = identity matrix
        if self.verbose:
            logger.info("Step 1: GMM with identity matrix.")

        W = np.eye(self._stack_moments(theta0).size)
        res1 = minimize(
            self._gmm_objective_fn,
            theta0,
            args=(W,),
            method=method,
            options=outer_options or {"disp": self.verbose},
            bounds=[(0.1, np.inf)],
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
            method=method,
            options=outer_options or {"disp": self.verbose},
            bounds=[(0, np.inf)],
        )
        theta2 = res2.x
        g2 = self._stack_moments(theta2)

        # save results
        self.theta = theta2
        self.W = W_opt
        self.g = g2
        self.est_params["beta_st"] = theta2[0]
        self.est_params["zeta"] = theta2[1]

        # Get other parameters
        self._labor_demand_parameters()
        self._amenity_supply_parameters()

        # Compute VCV matrix ?
        logger.info("GMM estimation finished.")

    def run_regulation_counterfactual(self):
        """
        Run a counterfactual simulation with a regulation shock to one city.
        """

        df = self.data

        # ---------------------------------------------------------
        # Select city with highest exogenous amenity
        # ---------------------------------------------------------
        # Get exogenous amenity
        delta_hat = self._blp_inversion(self.est_params["beta_st"])
        a = []
        for race in self.params["race_types"]:
            _, _, _, amenity_exog = self._estimate_2sls(
                delta_hat, self.data, self.est_params["zeta"], race=race
            )
            amenity_exog_H = amenity_exog[: len(self.data)]
            amenity_exog_L = amenity_exog[len(self.data) :]

            a.append(amenity_exog_H / (np.std(amenity_exog_H)))
            a.append(amenity_exog_L / (np.std(amenity_exog_L)))

        a = np.mean(np.array(a), axis=0)
        df["exog_amenity"] = a
        # var = "Z_H"
        x_axis_var = "exog_amenity"

        # Get city with highest exogenous amenity
        city = df.loc[df[x_axis_var].idxmax()]["City"]
        # print(f"City with highest {x_axis_var}: {city}")

        # ----------------------------------------------------------
        # Update parameters
        # ----------------------------------------------------------

        # Create shock to max city and 0 to all other cities
        regulation_shock = np.zeros(len(self.data))
        regulation_shock[df["City"] == city] = 5

        params = copy.deepcopy(self.DD.params)
        for key in self.est_params:
            val = self.est_params[key]
            if type(val) == dict:
                for sub_key in val:
                    sub_val = val[sub_key]
                    params[key][sub_key] = sub_val
            else:
                params[key] = val

        self.print_results()
        # ---------------------------------------------------------
        # Simulate data with new parameters
        # ---------------------------------------------------------

        def resimulate_data(update_params=False):
            DD = DiamondData(copy.deepcopy(self.DD.params), seed=self.DD.seed)
            DD._simulate_exog()
            DD.x_reg = DD.x_reg + regulation_shock
            if update_params:
                DD.params = params
            DD.phi = (
                DD.params["phi"]
                + DD.params["phi_geo"] * DD.x_geo
                + DD.params["phi_reg"] * DD.x_reg
            )
            DD._simulate_endog()
            return DD

        DD_new = resimulate_data(update_params=True)
        DD_old = resimulate_data(update_params=False)

        DD_new_params = DD_new.to_dataframe()
        DD_old_params = DD_old.to_dataframe()
        DD_preshock = df.copy()

        DD_new_params = DD_new_params[DD_new_params.City != city]
        DD_old_params = DD_old_params[DD_old_params.City != city]
        DD_preshock = DD_preshock[DD_preshock.City != city]

        vars = ["High_Ed_Population", "Low_Ed_Population", "Log_Rent"]
        labels = ["High Skill Population", "Low Skill Population", "Rent"]

        # x_axis_label = "High Skill Demand Shock"
        x_axis_label = "Exogenous Amenity"

        # Plot difference
        for i, var in enumerate(vars):

            x_axis = DD_preshock[x_axis_var].to_numpy()

            diff_old = DD_old_params[var] - DD_preshock[var]
            diff_new = DD_new_params[var] - DD_preshock[var]

            plt.figure(figsize=(10, 6))
            plt.scatter(x_axis, diff_old, alpha=0.5, color="blue", label="True Params")
            plt.scatter(
                x_axis,
                diff_new,
                alpha=0.5,
                color="red",
                label="Estimated Params",
            )

            # Plot best fit line
            z = np.polyfit(x_axis, diff_old, 1)
            p = np.poly1d(z)
            plt.plot(x_axis, p(x_axis), color="blue")

            z = np.polyfit(x_axis, diff_new, 1)
            p = np.poly1d(z)
            plt.plot(x_axis, p(x_axis), color="red")

            plt.xlabel(x_axis_label)
            plt.ylabel(f"Change in {labels[i]}")
            plt.title(f"Change in {labels[i]} After Regulatory Constraint Shock")
            plt.grid()
            plt.legend()
            plt.show()

    def run_amenity_counterfactual(self, output_folder):
        """
        Run a counterfactual simulation where endogenous amenity is fixed.
        """

        # ----------------------------------------------------------
        # Update parameters
        # ----------------------------------------------------------
        params = copy.deepcopy(self.DD.params)
        for key in self.est_params:
            val = self.est_params[key]
            if type(val) == dict:
                for sub_key in val:
                    sub_val = val[sub_key]
                    params[key][sub_key] = sub_val
            else:
                params[key] = val

        # ---------------------------------------------------------
        # Simulate data with new parameters
        # ---------------------------------------------------------
        def resimulate_data(update_params=False):

            if update_params:
                DD = DiamondData(params, seed=self.DD.seed)
            else:
                DD = DiamondData(copy.deepcopy(self.DD.params), seed=self.DD.seed)

            # Shut down endogenous amenity
            DD.params["phi_a"] = 0
            DD.simulate()
            return DD

        DD_update = resimulate_data(update_params=True).to_dataframe()
        # DD_original = resimulate_data(update_params=False).to_dataframe()
        DD_endog_amenity = self.data

        DD_update["Wage_Gap"] = DD_update["Log_Wage_H"] - DD_update["Log_Wage_L"]
        DD_endog_amenity["Wage_Gap"] = (
            DD_endog_amenity["Log_Wage_H"] - DD_endog_amenity["Log_Wage_L"]
        )

        vars = ["Log_Wage_H", "Log_Wage_L", "Log_H", "Log_L", "Log_Rent"]
        labels = [
            "High Skill Wage",
            "Low Skill Wage",
            "High Skill Pop",
            "Low Skill Pop",
            "Log Rent",
        ]

        shocks = ["Z_H", "Z_L"]
        labels = [
            "High Skill Demand Shock",
            "Low Skill Demand Shock",
        ]

        for i, shock in enumerate(shocks):
            y_axis_var_labels = [
                "Log High Ed Population",
                "Log Low Ed Population",
                "Log Rent",
            ]

            for j, y_axis_var in enumerate(["Log_H", "Log_L", "Log_Rent"]):
                x_axis = np.log(DD_endog_amenity[shock].to_numpy())
                var_update = DD_update[y_axis_var]
                var_amenity = DD_endog_amenity[y_axis_var]

                plt.figure(figsize=(10, 6))
                plt.scatter(
                    x_axis,
                    var_amenity,
                    alpha=0.5,
                    color="blue",
                    label="Endogenous Amenity",
                )
                plt.scatter(
                    x_axis,
                    var_update,
                    alpha=0.5,
                    color="red",
                    label="Exogenous Amenity",
                )

                # Plot best fit line
                z = np.polyfit(x_axis, var_amenity, 1)
                p = np.poly1d(z)
                plt.plot(x_axis, p(x_axis), color="blue")

                z = np.polyfit(x_axis, var_update, 1)
                p = np.poly1d(z)
                plt.plot(x_axis, p(x_axis), color="red")

                plt.xlabel(f"{labels[i]}")
                plt.ylabel(y_axis_var_labels[j])
                plt.grid()
                plt.legend()
                plt.savefig(
                    f"{output_folder}/scatter_{labels[i]}_{y_axis_var}.png",
                    bbox_inches="tight",
                    dpi=300,
                )
                plt.show()

        # Plot histogram of High_Ed_Share in DD_update vs DD_endog_amenity
        var_labels = [
            "High Skill Population",
            "Low Skill Population",
            "High Skill Share of Population",
            "Log Rent",
            "Endogenous Amenity",
            "Wage Gap",
        ]
        for i, var in enumerate(
            [
                "High_Ed_Population",
                "Low_Ed_Population",
                "High_Ed_Share",
                "Log_Rent",
                "Amenity_Endog",
                "Wage_Gap",
            ]
        ):
            plt.figure(figsize=(10, 6))
            plt.hist(
                DD_update[var],
                bins=30,
                alpha=0.5,
                color="red",
                label="Exogenous Amenity",
                density=True,
            )
            plt.hist(
                DD_endog_amenity[var],
                bins=30,
                alpha=0.5,
                color="blue",
                label="Endogenous Amenity",
                density=True,
            )
            plt.xlabel(f"{var_labels[i]}")
            plt.ylabel("Frequency")
            plt.legend()
            plt.grid()
            plt.savefig(
                f"{output_folder}/histogram_{var}.png", bbox_inches="tight", dpi=300
            )
            plt.show()

        # Plot difference
        # for i, var in enumerate(vars):

        #     x_axis = DD_endog_amenity[var].to_numpy()

        #     var_update = DD_update[var]
        #     var_original = DD_original[var]
        #     var_amenity = DD_endog_amenity[var]

        #     plt.figure(figsize=(10, 6))
        #     # plt.scatter(
        #     #     x_axis, var_original, alpha=0.5, color="blue", label="True Params"
        #     # )
        #     plt.scatter(
        #         x_axis,
        #         var_update,
        #         alpha=0.5,
        #         color="red",
        #         label="Estimated Params",
        #     )
        #     # plt.scatter(
        #     #     x_axis,
        #     #     var_amenity,
        #     #     alpha=0.5,
        #     #     color="green",
        #     #     label="Endogenous Amenity",
        #     # )

        #     # Plot best fit line
        #     # z = np.polyfit(x_axis, var_original, 1)
        #     # p = np.poly1d(z)
        #     # plt.plot(x_axis, p(x_axis), color="blue")

        #     z = np.polyfit(x_axis, var_update, 1)
        #     p = np.poly1d(z)
        #     plt.plot(x_axis, p(x_axis), color="red")

        #     # z = np.polyfit(x_axis, var_amenity, 1)
        #     # p = np.poly1d(z)
        #     # plt.plot(x_axis, p(x_axis), color="green")

        #     # Plot 45 degree line
        #     plt.plot(x_axis, x_axis, color="black", linestyle="-")

        #     plt.xlabel(f"{labels[i]} (endog)")
        #     plt.ylabel(f"{labels[i]}")
        #     plt.grid()
        #     plt.legend()
        #     plt.show()

    def print_results(self, est_params=None):

        if est_params is None:
            est_params = self.est_params

        print("=== Estimation Results ===")
        for key in est_params:
            val = est_params[key]
            if type(val) == dict:
                for sub_key in val:
                    sub_val = val[sub_key]
                    sub_val_true = self.params[key][sub_key]
                    print(
                        f"{key}[{sub_key}] = {sub_val:.4f} (true: {sub_val_true:.4f})"
                    )
            else:
                val_true = self.params[key]
                print(f"{key} = {val:.4f} (true: {val_true:.4f})")
