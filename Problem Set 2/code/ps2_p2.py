# %%

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import toml

from scipy.optimize import minimize

# %%

with open("params_q2.toml", "r") as file:
    params = toml.load(file)
print(params)


# %%

np.random.seed(14_273)


def process_data(dir="../data/"):
    # 1. Load the data
    entry = pd.read_csv(os.path.join(dir, "entryData.csv"), header=None)
    entry.rename(
        columns={
            0: "X",
            1: "Z_1",
            2: "Z_2",
            3: "Z_3",
            4: "enter_1",
            5: "enter_2",
            6: "enter_3",
        },
        errors="raise",
        inplace=True,
    )
    entry = entry.assign(market=range(len(entry)))
    # reshape
    entry = pd.wide_to_long(entry, ["Z", "enter"], i="market", j="firm", sep="_")
    entry = entry.reset_index().sort_values(["market", "firm"])
    return entry


entry = process_data()


def make_shock_matrix(data, N_draws=params["N_draws"]):
    shocks = np.random.normal(0, 1, size=(len(data), N_draws))
    return shocks


shocks = make_shock_matrix(entry)


# %%
def get_simulated_prob_entry(theta, data, shocks):
    mu = theta[0]
    sigma = theta[1]
    alpha, beta, delta, N_draws = (
        params["alpha"],
        params["beta"],
        params["delta"],
        params["N_draws"],
    )
    tmp_data = data.copy()
    tmp_data["constant_profit_component"] = tmp_data["X"] * beta
    res = []
    for i in range(N_draws):
        # calculate firm specific phi
        tmp_data["phi_fm"] = tmp_data["Z"] * alpha + sigma * (mu + shocks[:, i])
        tmp_data = tmp_data.sort_values(by=["market", "phi_fm"], ascending=[True, True])
        tmp_data["firm_rank"] = tmp_data.groupby("market").cumcount() + 1
        # estimate profits from entering and determine entering decision
        tmp_data["profits_if_enter"] = (
            tmp_data["constant_profit_component"]
            - delta * np.log(tmp_data["firm_rank"])
            - tmp_data["phi_fm"]
        )
        prediction = (tmp_data["profits_if_enter"] > 0).astype(int)
        res.append(prediction.rename(f"predicted_entry_{i}"))
    # combine all entry predictions
    preds_all = pd.concat(res, axis=1)
    preds = preds_all.mean(axis=1).rename("predicted_entry_mean")
    data = pd.concat((data, preds), axis=1)
    # calculate the log likelihood (1e-16 is a small constant to avoid log(0))
    data["llh"] = np.log(data["predicted_entry_mean"] + 1e-16) * data["enter"] + np.log(
        1 - data["predicted_entry_mean"] + 1e-16
    ) * (1 - data["enter"])
    llh = -np.mean(data["llh"])
    return llh


results = minimize(
    get_simulated_prob_entry,
    [2, 1],
    args=(entry, shocks),
    # BW NOTE: i think we can change this to like 1e-6
    tol=1e-15,
    method="Nelder-Mead",
    bounds=[(None, None), (0.1, None)],
)


# %%
def sim_likelihood(
    data,
    mu,
    sigma,
    N_draws=params["N_draws"],
    alpha=params["alpha"],
    beta=params["beta"],
):
    # make spine
    spine = data[["market", "firm"]]
    datasets = []
    for i in range(N_draws):
        datasets = datasets.append(i)
        i = spine.copy()
        "draw{0}".format(i)["predict{0}".format(i)] = get_simulated_prob_entry(
            data, mu, sigma, alpha, beta
        )
    return data, datasets


entry = sim_likelihood(entry, mu=0, sigma=1)

# %%
