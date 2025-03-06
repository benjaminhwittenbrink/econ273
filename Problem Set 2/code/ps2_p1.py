import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt


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
    K = len(regs)
    # Output GM Table
    table = pd.DataFrame(
        {
            "Column": [
                "Balanced, Total",
                "Balanced, Within",
                "Full, Total",
                "Full, + Investment",
            ],
            "Log employment": [regs[i].params["lemp"] for i in range(K)],
            "Log capital": [regs[i].params["ldnpt"] for i in range(K)],
            "Log R&D capital": [regs[i].params["ldrst"] for i in range(K)],
            "Log investment": [None, None, None, regs[3].params["ldinv"]],
            "N": [regs[i].nobs for i in range(K)],
            "R^2": [regs[i].rsquared for i in range(K)],
        }
    )
    return table
