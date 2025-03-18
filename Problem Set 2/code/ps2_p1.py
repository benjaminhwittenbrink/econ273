import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

from scipy.optimize import minimize


def load_data(dir="../data/"):
    # 1. Load the data
    gmd = pd.read_csv(os.path.join(dir, "GMdata.csv"))
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

    col1_f = "ldsal ~ 0 + lemp + ldnpt + ldrst + C(yr) + C(yr):sic_357"
    col_fs = [
        col1_f,  # Column (1): Balanced, Total
        col1_f + " + C(index)",  # Column (2): Balanced, Within
        col1_f,  # Column (3): Full, Total
        col1_f + " + ldinv",  # Column (4): Full, + Investment
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
        ("Log employment", "lemp"),
        ("Log capital", "ldnpt"),
        ("Log R&D capital", "ldrst"),
        ("Log investment", "ldinv"),
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
def acf_rho_diff(df):
    df_lag = (
        df.sort_values(["index", "yr"])
        .groupby("index")
        .apply(
            lambda g: g.assign(
                ldsal_lag=lambda x: x["ldsal"].shift(1),
                lemp_lag=lambda x: x["lemp"].shift(1),
                ldnpt_lag=lambda x: x["ldnpt"].shift(1),
                ldrst_lag=lambda x: x["ldrst"].shift(1),
            ),
        )
        .reset_index(drop=True)
        .dropna(subset=["ldsal_lag"])
    )

    rho_init = 0.5
    result = minimize(
        lambda r: rho_gmm_objetive(df_lag, r),
        x0=rho_init,
        method="L-BFGS-B",
    )
    rho_hat = result.x[0]
    beta_hat, _ = rho_diff_moments(df_lag, rho_hat)
    return {
        "rho": rho_hat,
        "beta_hat": beta_hat,
        "success": result.success,
        "message": result.message,
    }


def rho_diff_moments(df, rho):
    # calculate rho differences variables
    df_diff = df.assign(
        ldsal_diff=lambda x: x["ldsal"] - rho * x["ldsal_lag"],
        lemp_diff=lambda x: x["lemp"] - rho * x["lemp_lag"],
        ldnpt_diff=lambda x: x["ldnpt"] - rho * x["ldnpt_lag"],
        ldrst_diff=lambda x: x["ldrst"] - rho * x["ldrst_lag"],
    )
    # NOTE: @TODO: need to do IV here
    X = df_diff[["lemp_diff", "ldnpt_diff", "ldrst_diff"]]
    y = df_diff["ldsal_diff"]
    mod = sm.OLS(y, sm.add_constant(X)).fit()
    return mod.params, mod.resid


def rho_gmm_objetive(df, rho):
    params, resid = rho_diff_moments(df, rho)
    moment_val = np.mean(df["lemp_lag"] * resid)
    return moment_val**2
