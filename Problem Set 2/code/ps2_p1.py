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
        self._demean_by_fixed_effects()

        # Create lags
        self._create_lags()

    def _demean_by_fixed_effects(
        self, cols=["sale", "emp", "capital", "rnd", "invest"]
    ):
        for col in cols:
            res = sm.OLS(self.df[col], self.df[self.fe_cols]).fit()
            self.df[f"{col}_raw"] = self.df[col]
            self.df[col] = self.df[col] - res.fittedvalues

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

    def _create_lag_df(self):
        # create t -1 lags in data
        self.df_lag = (
            self.df.sort_values(["index", "yr_cat"])
            .groupby("index")
            .apply(
                lambda g: g.assign(
                    sale_lag=lambda x: x["sale"].shift(1),
                    emp_lag=lambda x: x["emp"].shift(1),
                    capital_lag=lambda x: x["capital"].shift(1),
                    rnd_lag=lambda x: x["rnd"].shift(1),
                    invest_lag=lambda x: x["invest"].shift(1),
                ),
            )
            .reset_index(drop=True)
        )

    # RHO PANEL DIFFERENCING METHODS
    def est_rho_diff_model(self, rho_init=None):
        # create lagged data
        self._create_lag_df()

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
        df_diff, fe_cols = self._rho_diff_process_data(rho)
        # create matrices for linear GMM
        X, Z, y = self._rho_diff_create_mat(df_diff, fe_cols)
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
        dat = self.df_lag.copy()
        dat = dat.assign(
            sic_357=lambda x: (x["sic3"] == 357).astype(int),
            yr_sic357=lambda x: x["yr"].astype(str) + "_" + x["sic_357"].astype(str),
        )
        dat = pd.get_dummies(dat, columns=["yr", "yr_sic357"], drop_first=False)
        dat = dat.drop(
            columns=[col for col in dat.columns if re.match(r"^yr_sic357_.*_0$", col)]
        )
        # calculate rho differences variables
        df_diff = dat.assign(
            sale_diff=lambda x: x["sale"] - rho * x["sale_lag"],
            emp_diff=lambda x: x["emp"] - rho * x["emp_lag"],
            capital_diff=lambda x: x["capital"] - rho * x["capital_lag"],
            rnd_diff=lambda x: x["rnd"] - rho * x["rnd_lag"],
        )
        fe_cols = sorted([col for col in dat.columns if col.startswith("yr_")])
        # For each subsequent year column, subtract rho times the previous column in-place
        for i in range(1, len(fe_cols)):
            current_col = fe_cols[i]
            prev_col = fe_cols[i - 1]
            df_diff[current_col + "_diff"] = (
                df_diff[current_col] - rho * df_diff[prev_col]
            )
        # drop rows where yr_73 is 1 (i.e. first year)
        df_diff = df_diff.query("yr_73 != 1")
        return df_diff, fe_cols

    def _rho_diff_create_mat(self, df, fe_cols):
        X = (
            df[
                [
                    "emp_diff",
                    "capital_diff",
                    "rnd_diff",
                ]
                + [col for col in fe_cols if "_73" not in col]
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
                + [col for col in fe_cols if "_73" not in col]
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
        phi = self._first_stage_fit_phi_poly(degree=degree)
        # X = sm.add_constant(phi)
        y = self.df["sale"].values
        res = sm.OLS(y, phi).fit()  # TODO Figure out what to do with fixed effects

        self.df["phi"] = res.fittedvalues
        # self.df["phi_resid"] = res.resid
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
            df.residuals, sm.add_constant(df.residuals_lag), missing="drop"
        ).fit()
        rho = res.params["residuals_lag"]
        mu = res.params["const"]
        xi = np.array(res.resid).T
        return rho, mu, xi

    def _second_stage_instruments(self):
        return self.df[["emp_lag", "capital", "rnd"]].dropna().to_numpy()

    def _second_stage_objective(self, params, W=np.eye(3)):
        # Moment condition
        rho, mu, xi = self._estimate_rho_mu(params)
        return (self.Z.T.dot(xi)).T @ W @ (self.Z.T.dot(xi)) * 1 / len(xi)

    def _second_stage_optimal_weights(self, xi):
        mat = (self.Z.T * (xi**2)) @ self.Z
        return np.linalg.pinv(mat / len(xi))

    def est_second_stage(self, num_moments=3):
        # Construct instruments
        self.Z = self._second_stage_instruments()

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
