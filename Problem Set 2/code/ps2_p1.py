import os
import re
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from linearmodels.iv import IV2SLS
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

from scipy.optimize import minimize


def load_data(dir="../data/"):
    # 1. Load the data
    gmd = pd.read_csv(os.path.join(dir, "GMdata.csv"))

    # Rename variables
    rename_dict = {
        "ldsal": "sale",
        "lemp": "emp",
        "ldnpt": "capital",
        "ldrst": "rnd",
        "ldinv": "invest",
    }
    gmd = gmd.rename(columns=rename_dict)

    return gmd


def run_reg(df, formula, **kwargs):
    return smf.ols(formula=formula, data=df, **kwargs).fit()


def estimate_markups(labor_elas):
    gmd = load_data()

    wage_data = pd.read_csv("../data/sic5811.csv")
    wage_data["sic3"] = wage_data["sic"].astype(str).str[:3].astype(int)
    wage_data["yr"] = wage_data["year"].astype(str).str[-2:].astype(int)
    wage_data["wage"] = 1000 * wage_data["pay"] / wage_data["emp"]

    # Collapse at sic3/year level taking weighted average with weights being total employment
    wage_data["wage"] = wage_data["wage"] * wage_data["emp"]
    wage_data = (
        (
            wage_data.groupby(["yr", "sic3"])["wage"].sum()
            / wage_data.groupby(["yr", "sic3"])["emp"].sum()
        )
        .reset_index()
        .rename(columns={0: "wage"})
    )

    # wage_data = wage_data.groupby(["sic3", "yr"])["pay"].mean().reset_index()

    df = gmd.merge(wage_data, on=["sic3", "yr"], how="left")

    df["markup"] = labor_elas / (
        (df["wage"] * 1000 * np.exp(df["emp"])) / (1e6 * np.exp(df["sale"]))
    )

    # Take average markup
    markups = df.groupby("yr")["markup"].mean()

    # Take weighted average markup, weighing by total sales
    df["weighted_markup"] = df["markup"] * df["sale"]
    weighted_markups = (
        df.groupby("yr")["weighted_markup"].sum() / df.groupby("yr")["sale"].sum()
    )

    # %% Plot markup timeseries
    fig, ax = plt.subplots()
    ax.plot(markups.index, markups, label="Unweighted Average")
    ax.plot(
        weighted_markups.index,
        weighted_markups,
        label="Weighted Average",
    )
    ax.legend()
    ax.set_xlabel("Year")
    ax.set_ylabel("Markup")
    plt.savefig("../output/markup_timeseries.png")


def replicate_GM(df):
    # process data
    df = df.assign(
        sic_357=lambda x: (x["sic3"] == 357).astype(int),
    )
    # create balanced panel
    df_bal = df.groupby(["index"]).filter(lambda x: len(x) == df["yr"].nunique())

    col1_f = "sale ~ 0 + emp + capital + rnd + C(yr) + C(yr):sic_357"
    col_fs = [
        col1_f,  # Column (1): Balanced, Total
        col1_f + " + C(index)",  # Column (2): Balanced, Within
        col1_f,  # Column (3): Full, Total
        col1_f + " + invest",  # Column (4): Full, + Investment
    ]
    regs = [None] * len(col_fs)
    for i, f in enumerate(col_fs):
        regs[i] = run_reg(df_bal if i < 2 else df, formula=f)
    return regs


def output_GM_table(regs):
    # Define the column labels for each regression.
    col_labels = [
        "Balanced, Total",
        "Balanced, Within",
        "Full, Total",
        "Full, + Investment",
    ]

    # Define the variable names and corresponding regression parameter keys.
    variables = [
        ("Log employment", "emp"),
        ("Log capital", "capital"),
        ("Log R&D capital", "rnd"),
        ("Log investment", "invest"),
    ]

    # Create a list of row keys in the order we want:
    # For each variable, one for coefficient and one for its standard error (unique key).
    row_keys = []
    for label, _ in variables:
        row_keys.append(label)
        row_keys.append(label + "_se")
    row_keys.extend(["N", "R^2"])

    # Initialize a dictionary with these row keys.
    table_data = {key: [] for key in row_keys}

    # Populate the table_data dictionary for each regression.
    for i, reg in enumerate(regs):
        for var_label, param in variables:
            # Only the fourth regression (index 3) has Log investment.
            if var_label == "Log investment" and i != 3:
                table_data[var_label].append("")
                table_data[var_label + "_se"].append("")
            else:
                table_data[var_label].append(f"{reg.params[param]:.3f}")
                table_data[var_label + "_se"].append(f"({reg.bse[param]:.3f})")
        # Add N and R^2 values.
        table_data["N"].append(reg.nobs)
        table_data["R^2"].append(f"{reg.rsquared:.3f}")

    # Create the DataFrame and transpose so that columns are regressions.
    df = pd.DataFrame(table_data).T
    df.columns = col_labels

    # Replace row keys ending with '_se' with empty strings.
    df.index = ["" if key.endswith("_se") else key for key in row_keys]
    return df


## ACF functions
class ACF:

    def __init__(self, df):
        self.df = df
        self._process_data()

    def _process_data(self):

        # Create dummies
        self._add_fixed_effects()

        # De-mean variables by fixed effects
        # self._demean_by_fixed_effects()

        # Create lags
        self._create_lags()

    def _demean_by_fixed_effects(self, df):

        cols = set()
        for t in ["sale", "emp", "capital", "rnd", "invest"]:
            for col in df.columns:
                if t in col:
                    cols.add(col)
        cols = sorted(list(cols))

        for col in cols:
            res = sm.OLS(df[col], df[self.fe_cols]).fit()
            df[f"{col}_raw"] = df[col]
            df[col] = df[col] - res.fittedvalues
        return df

    def _add_fixed_effects(self):
        self.df = self.df.assign(
            yr_cat=lambda x: x["yr"],
            sic_357=lambda x: (x["sic3"] == 357).astype(int),
            yr_sic357=lambda x: x["yr"].astype(str) + "_" + x["sic_357"].astype(str),
        )
        self.df = pd.get_dummies(
            self.df,
            columns=["yr", "yr_sic357"],
            drop_first=False,
        )
        self.df = self.df.drop(
            columns=[
                col for col in self.df.columns if re.match(r"^yr_sic357_.*_0$", col)
            ]
        )

        self.fe_cols = sorted(
            [
                col
                for col in self.df.columns
                if col.startswith("yr_") and col != "yr_cat"
            ]
        )

        # Convert to zero-one variable
        for col in self.fe_cols:
            self.df[col] = self.df[col].astype(int)

    def _create_lags(self, columns=["sale", "emp", "capital", "rnd", "invest"]):
        self.df = self.df.sort_values(["index", "yr_cat"])
        for col in columns:
            self.df[col + "_lag"] = self.df[col].groupby(self.df["index"]).shift(1)

    # RHO PANEL DIFFERENCING METHODS
    def est_rho_diff_model(self, rho_init=None):

        if rho_init is None:
            rho_init = 0.5
        result = minimize(
            self._rho_diff_objective,
            x0=rho_init,
            bounds=[(0, 1)],
            method="L-BFGS-B",
        )
        rho_hat = result.x[0]
        _, beta_hat = self._rho_diff_objective(rho_hat, return_params=True)
        return {
            "rho": rho_hat,
            "beta_hat": beta_hat,
            "success": result.success,
            "message": result.message,
        }

    def _rho_diff_objective(self, rho, return_params=False):
        print("Minimizing rho: ", rho)
        # create rho differenced data
        df = self._rho_diff_process_data(rho)
        # create matrices for linear GMM
        X, Z, y = self._rho_diff_create_mat(df)
        # first stage with W = I
        W = np.identity(Z.shape[1])
        beta_1 = self._rho_diff_linear_gmm(X, Z, W, y)
        resid = y - X.dot(beta_1)
        # second stage with optimal W
        W = self._rho_diff_optimal_weights(Z, resid)
        beta_2 = self._rho_diff_linear_gmm(X, Z, W, y)
        resid_2 = y - X.dot(beta_2)
        # calculate GMM objective
        resid_2_mat = np.reshape(resid_2, (-1, 1))
        moment_val = (Z.T.dot(resid_2_mat)).T @ W @ (Z.T.dot(resid_2_mat))
        if return_params:
            return moment_val, beta_2
        return moment_val

    def _rho_diff_process_data(self, rho):
        # create year fixed effects
        df = self.df.copy()
        df = df.assign(
            sale_diff=lambda x: x["sale"] - rho * x["sale_lag"],
            emp_diff=lambda x: x["emp"] - rho * x["emp_lag"],
            capital_diff=lambda x: x["capital"] - rho * x["capital_lag"],
            rnd_diff=lambda x: x["rnd"] - rho * x["rnd_lag"],
        )

        # Get double lag for employment
        # df["emp_lag2"] = self.df["emp"].groupby(self.df["index"]).shift(2)

        # drop rows where missing lag
        df = df.dropna()
        df = self._demean_by_fixed_effects(df)
        return df

    def _rho_diff_create_mat(self, df):
        X = (
            df[
                [
                    "emp_diff",
                    "capital_diff",
                    "rnd_diff",
                ]
                # + [col for col in self.fe_cols]
            ]
            .dropna()
            .astype(float)
        )
        Z = (
            df.loc[X.index][
                [
                    "invest_lag",
                    "emp_lag",
                    "capital_lag",
                    "rnd_lag",
                ]
                # + [col for col in self.fe_cols]
            ]
            .astype(float)
            .values
        )
        y = df.loc[X.index]["sale_diff"].values
        X = X.values
        return X, Z, y

    def _rho_diff_linear_gmm(self, X, Z, W, y):
        # linear gmm = (X' Z W Z' X)^{-1} (X' Z W Z' y)
        term1 = X.T.dot(Z).dot(W).dot(Z.T).dot(X)
        term1_inv = np.linalg.pinv(term1)
        term2 = X.T.dot(Z).dot(W).dot(Z.T).dot(y)
        return term1_inv.dot(term2)

    def _rho_diff_optimal_weights(self, Z, resid):
        mat = (Z.T * (resid.flatten() ** 2)) @ Z
        return np.linalg.pinv(mat / Z.shape[0])

    # ACF FIRST STAGE
    def est_first_stage(self, degree=2):
        #### WITHOUT FE
        # phi = self._first_stage_fit_phi_poly(degree=degree)
        # # X = sm.add_constant(phi)
        # y = self.df["sale"].values
        # res = sm.OLS(y, phi).fit()

        # self.df["phi"] = res.fittedvalues
        # # self.df["phi_resid"] = res.resid

        ###### WITH FE
        phi_poly = self._first_stage_fit_phi_poly(degree=degree)
        # Retrieve fixed effects dummies
        fe = self.df[self.fe_cols].values
        # Stack polynomial features and fixed effects dummies together.
        X = np.hstack([phi_poly, fe])
        y = self.df["sale"].values
        res = sm.OLS(y, X).fit()
        # Store the full fitted value (which includes the fixed-effects part)
        self.df["phi_full"] = res.fittedvalues

        # Remove the estimated fixed effects contribution.
        # The estimated coefficients are in the order: constant, coefficients for phi_poly,
        # and then coefficients for the fixed effects dummies.
        k = phi_poly.shape[1]  # number of polynomial terms
        # Get the coefficients for the fixed effects
        fe_coefs = res.params[k:]  # skip constant and phi_poly coefficients
        fe_effect = (
            fe @ fe_coefs
        )  # compute the fixed effects contribution for each observation
        # Define phi as the part due solely to the polynomial regressors.
        self.df["phi"] = self.df["phi_full"] - fe_effect

        return res

    def _first_stage_fit_phi_poly(self, degree=3):
        poly = PolynomialFeatures(degree=degree)
        phi = poly.fit_transform(self.df[["emp", "capital", "rnd", "invest"]])
        return phi

    def _estimate_rho_mu(self, params):
        beta_1, beta_2, beta_3 = params
        df = self.df

        # Calculate residual (Phi - known params)
        df["residuals"] = (
            df["phi"] - beta_1 * df["emp"] - beta_2 * df["capital"] - beta_3 * df["rnd"]
        )

        # Take lag of residual
        df = df.sort_values(["index", "yr_cat"])
        df["residuals_lag"] = df["residuals"].groupby(df["index"]).shift(1)

        res = sm.OLS(
            df.residuals,
            sm.add_constant(df[["residuals_lag"]]),
            missing="drop",
        ).fit()
        rho = res.params["residuals_lag"]
        mu = res.params["const"]
        xi = np.array(res.resid).T
        return rho, mu, xi

    def _second_stage_instruments(self):
        return self.df.dropna()[["emp_lag", "capital", "rnd"]].to_numpy()

    def _second_stage_objective(self, params, W=np.eye(3)):
        # Moment condition
        rho, mu, xi = self._estimate_rho_mu(params)
        return (self.Z.T.dot(xi)).T @ W @ (self.Z.T.dot(xi)) * 1 / len(xi)

    def _second_stage_optimal_weights(self, xi):
        mat = (self.Z.T * (xi**2)) @ self.Z
        return np.linalg.pinv(mat / len(xi))

    def est_second_stage(self):
        # Construct instruments
        self.Z = self._second_stage_instruments()
        num_moments = self.Z.shape[1]

        # GMM
        # Stage 1
        W = np.eye(num_moments)
        params_init = [1, 1, 1]
        res = minimize(
            self._second_stage_objective, params_init, args=(W,), method="L-BFGS-B"
        )

        # Calculate optimal weights
        rho, mu, xi = self._estimate_rho_mu(res.x)
        W = self._second_stage_optimal_weights(xi)

        # Stage 3
        res = minimize(
            self._second_stage_objective, res.x, args=(W,), method="L-BFGS-B"
        )

        rho, mu, xi = self._estimate_rho_mu(res.x)

        return [rho, mu] + list(res.x)

    def _second_stage_optimal_weights_alt(self, xi_epsilon):
        mat = (self.Z.T * (xi_epsilon**2)) @ self.Z
        return np.linalg.pinv(mat / len(xi_epsilon))

    def _estimate_xi_epsilon(self, params):
        rho, beta_1, beta_2, beta_3 = params
        df = self.df.dropna()

        xi_epsilon = df["sale"] - (
            beta_1 * df["emp"]
            + beta_2 * df["capital"]
            + beta_3 * df["rnd"]
            + rho
            * (
                df["phi_lag"]
                - beta_1 * df["emp_lag"]
                - beta_2 * df["capital_lag"]
                - beta_3 * df["rnd_lag"]
            )
        )

        # Residualize on fixed effects
        res = sm.OLS(xi_epsilon, df[self.fe_cols]).fit()
        xi_epsilon = np.array(res.resid).T
        return xi_epsilon

    def _second_stage_objective_alt(self, params, W=np.eye(3)):
        xi_epsilon = self._estimate_xi_epsilon(params)

        loss = (
            (self.Z.T.dot(xi_epsilon)).T
            @ W
            @ (self.Z.T.dot(xi_epsilon))
            * 1
            / len(xi_epsilon)
        )
        # print(params)
        return loss

    def _second_stage_instruments_alt(self):
        return sm.add_constant(
            self.df.dropna()[["emp_lag", "capital_lag", "rnd_lag", "phi_lag"]]
        ).to_numpy()

    def est_second_stage_alt(self):
        # Construct instruments

        self.df["phi_lag"] = self.df["phi"].groupby(self.df["index"]).shift(1)
        self.Z = self._second_stage_instruments_alt()

        num_moments = self.Z.shape[1]

        # GMM
        # Stage 1
        W = np.eye(num_moments)
        params_init = [0.5, 0.1, 0.3, 0.5]  # rho + betas
        res = minimize(
            self._second_stage_objective_alt,
            params_init,
            args=(W,),
            method="L-BFGS-B",
            bounds=[(0.00001, None)] * len(params_init),
        )

        # Calculate optimal weights
        xi_epsilon = self._estimate_xi_epsilon(res.x)
        W = self._second_stage_optimal_weights_alt(xi_epsilon)

        # Stage 3
        res = minimize(
            self._second_stage_objective_alt,
            res.x,
            args=(W,),
            method="L-BFGS-B",
            bounds=[(0.00001, None)] * len(params_init),
        )
        return res.x

    def _estimate_xi_epsilon_survival_control(self, params):
        rho, beta_1, beta_2, beta_3, alpha = params
        df = self.df.dropna()

        xi_epsilon = df["sale"] - (
            beta_1 * df["emp"]
            + beta_2 * df["capital"]
            + beta_3 * df["rnd"]
            + rho
            * (
                df["phi_lag"]
                - beta_1 * df["emp_lag"]
                - beta_2 * df["capital_lag"]
                - beta_3 * df["rnd_lag"]
            )
            + alpha * df["capital_lag"]
        )

        # Residualize on fixed effects
        res = sm.OLS(xi_epsilon, df[self.fe_cols]).fit()
        xi_epsilon = np.array(res.resid).T

        return xi_epsilon

    def _second_stage_objective_survival_control(self, params, W=np.eye(3)):
        # Moment condition
        xi_epsilon = self._estimate_xi_epsilon_survival_control(params)
        loss = (
            (self.Z.T.dot(xi_epsilon)).T
            @ W
            @ (self.Z.T.dot(xi_epsilon))
            * 1
            / len(xi_epsilon)
        )
        # print(params)
        return loss

    def est_second_stage_survival_control(self):
        # Construct instruments

        self.df["phi_lag"] = self.df["phi"].groupby(self.df["index"]).shift(1)
        self.Z = self._second_stage_instruments_alt()

        num_moments = self.Z.shape[1]

        # GMM
        # Stage 1
        W = np.eye(num_moments)
        params_init = [0.5, 0.1, 0.3, 0.5, 0]  # rho + betas + alpha
        res = minimize(
            self._second_stage_objective_survival_control,
            params_init,
            args=(W,),
            method="L-BFGS-B",
            bounds=[(None, None)] * len(params_init),
        )

        # Calculate optimal weights
        xi_epsilon = self._estimate_xi_epsilon_survival_control(res.x)
        W = self._second_stage_optimal_weights_alt(xi_epsilon)

        # Stage 3
        res = minimize(
            self._second_stage_objective_survival_control,
            res.x,
            args=(W,),
            method="L-BFGS-B",
            bounds=[(None, None)] * len(params_init),
        )
        return res.x
