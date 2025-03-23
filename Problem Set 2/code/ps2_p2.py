# %%

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import toml
from statsmodels.discrete.discrete_model import MNLogit
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from joblib import Parallel, delayed
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
    return entry

def make_long(data):
    data = data.assign(market=range(len(data)))
    # reshape
    data_long = pd.wide_to_long(data, ["Z", "enter"], i="market", j="firm", sep="_")
    data_long = data_long.reset_index().sort_values(["market", "firm"])
    return data_long

def make_shock_matrix(data, N_draws=params["N_draws"], F = params["F"]):
    shocks = np.random.normal(0, 1, size=(len(data), N_draws))
    market = np.floor(np.array(range(len(shocks)))/F)
    shocks = np.column_stack((market, shocks))
    return shocks


entry = process_data()
entry_long = make_long(entry)
shocks = make_shock_matrix(entry_long)


# %%
def get_simulated_prob_entry(theta, data, shocks, params):
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
        tmp_data["phi_fm"] = tmp_data["Z"] * alpha + sigma * (mu + shocks[:, i+1])
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


results_Berry = minimize(
    get_simulated_prob_entry,
    [1, .5],
    args=(entry_long, shocks, params),
    # BW NOTE: i think we can change this to like 1e-6
    tol=1e-6,
    method="Nelder-Mead",
    bounds=[(None, None), (0.1, None)],
)


# %%
# get MNL predictions 
def get_mnl(data):
    data['entry_vector'] = data.enter_1.astype(str) + '_' + data.enter_2.astype(str) + '_' + data.enter_3.astype(str)
    regressors = pd.DataFrame(np.column_stack(( np.ones(len(entry)), entry.X , entry.Z_1, entry.Z_2,  entry.Z_3)))
    mnl = LogisticRegression(multi_class='multinomial').fit(regressors, entry.entry_vector)
    #get predictions
    predicted_outcomes_mnl = pd.DataFrame(mnl.predict_proba(regressors))
    #rename predicted to be the outomces
    predicted_outcomes_mnl.columns = mnl.classes_.tolist()
    predicted_outcomes_mnl.columns = 'T_' + predicted_outcomes_mnl.columns
    # fill in the remaining columns with 0 
    for i in range (2): 
        for j in range(2): 
            for k in range(2): 
                if f'T_{i}_{j}_{k}' not in predicted_outcomes_mnl.columns: 
                    predicted_outcomes_mnl[f'T_{i}_{j}_{k}'] = 0
    return predicted_outcomes_mnl

predicted_outcomes_mnl = get_mnl(entry)

def prep_for_mi(data, params): 
    beta = params["beta"]
    mi_data = data.copy()
    mi_data["constant_profit_component"] = mi_data["X"] * beta
    # store the actual equilibrium in the data 
    mi_data = mi_data.sort_values(by=["market", "firm"], ascending=[True, True])
    mi_data['enter_string']  =  mi_data.enter.astype(str) + '_'
    mi_data['entry_vector']  =  mi_data.groupby(['market'])['enter_string'].sum().astype(str)
    mi_data['entry_vector']  =  mi_data.groupby(['market'])['entry_vector'].ffill().str[:-1]
    return mi_data

mi_data = prep_for_mi(entry_long, params)
   
def get_mi(mu, data, predicted_outcomes_mnl, shocks, params):
    
    alpha, beta, delta, N_draws, sigma = (
        params["alpha"],
        params["beta"],
        params["delta"],
        params["N_draws"],
        params["sigma"],
    )
    tmp_data = data.copy()
    res_L = []
    res_H = []
    entry_combinations = [(i, j, k) for i in range(2) for j in range(2) for k in range(2)]
    # function that calculates equilibria for a given draw of n
    for n in range(N_draws):
        # calculate firm specific phi
        tmp_data["phi_fm"] = tmp_data["Z"] * alpha + sigma * (mu + shocks[:, n])
        # estimate profits from each possible entry and vector and determine if it's an equilibrium
        for i , j, k in entry_combinations:
            # get profits if this many firms enter 
                if i+j+k>0:   
                    tmp_data["profits_in_eqbm"] = i*(tmp_data['firm']==1)*(
                        tmp_data["constant_profit_component"]
                        - delta * np.log(i+j+k)
                        - tmp_data["phi_fm"]
                    )
                    tmp_data["profits_in_eqbm"] = tmp_data["profits_in_eqbm"] + j*(tmp_data['firm']==2)*(
                        tmp_data["constant_profit_component"]
                        - delta * np.log(i+j+k)
                        - tmp_data["phi_fm"]
                    )
                    tmp_data["profits_in_eqbm"] = tmp_data["profits_in_eqbm"] +  k*(tmp_data['firm']==3)*(
                        tmp_data["constant_profit_component"]
                        - delta * np.log(i+j+k)
                        - tmp_data["phi_fm"]
                    )
                    tmp_data = tmp_data.rename(columns={"profits_in_eqbm":f"profits_if_enter_{i}_{j}_{k}"})
                else: 
                    tmp_data["profits_if_enter_0_0_0"] = 0
        # determine which vectors are equilibria 
        for i , j, k in entry_combinations:
            # check for profitable deviation  
            i_comp = (i+1)%2
            j_comp = (j+1)%2
            k_comp = (k+1)%2
            tmp_data["profitable_deviation"] = (
            tmp_data.firm ==1 * (tmp_data["profits_if_enter_%s_%s_%s"%(i,j,k)] < tmp_data["profits_if_enter_%s_%s_%s"%(i_comp,j,k)])) | (
            tmp_data.firm ==2 * (tmp_data["profits_if_enter_%s_%s_%s"%(i,j,k)] < tmp_data["profits_if_enter_%s_%s_%s"%(i,j_comp,k)])) | (
            tmp_data.firm ==3 * (tmp_data["profits_if_enter_%s_%s_%s"%(i,j,k)] < tmp_data["profits_if_enter_%s_%s_%s"%(i,j,k_comp)])) 
            tmp_data["check_equilibrium"] = (tmp_data.groupby(["market"])["profitable_deviation"].max() == 0) 
            tmp_data["check_equilibrium"] = tmp_data.groupby(["market"])["check_equilibrium"].ffill()
            tmp_data = tmp_data.drop(columns = "profitable_deviation")
            tmp_data = tmp_data.rename(columns={"check_equilibrium":f"is_equilibrium_{i}_{j}_{k}_draw_{n}"})
        # check for unique equilibria 
        tmp_data["unique_equilibrium"] = tmp_data.filter(like="is_equilibrium_").sum(axis=1) == 1
        for i , j, k in entry_combinations:
            tmp_data["is_unique_equilibrium"] = tmp_data["is_equilibrium_%s_%s_%s_draw_%s" % (i,j,k,n)]* tmp_data["unique_equilibrium"]
            tmp_data = tmp_data.rename(columns={"is_unique_equilibrium":f"is_unique_equilibrium_{i}_{j}_{k}_draw_{n}"})

        # drop unnecessary columns
        tmp_data = tmp_data.drop(columns = list(tmp_data.filter(regex = "profits_if_enter_")))
        tmp_data = tmp_data.drop(columns = 'unique_equilibrium')
        # store predicted equilibria 
        prediction_H =  tmp_data.filter(like = "is_equilibrium_")[tmp_data['firm']==1].astype(int)
        res_H.append(prediction_H) 
        
        prediction_L =  tmp_data.filter(like = "is_unique_equilibrium_")[tmp_data['firm']==1].astype(int)
        res_L.append(prediction_L)
        tmp_data = tmp_data[tmp_data.columns.drop(list(tmp_data.filter(like = "_equilibrium_")))]
     
    # combine all entry predictions
    preds_all_H = pd.concat(res_H , axis=1)
    preds_H = pd.DataFrame()
    preds_L = pd.DataFrame()

    for i , j, k in entry_combinations:
        preds_H[f"H_{i}_{j}_{k}"] = preds_all_H.filter(like = f"is_equilibrium_{i}_{j}_{k}").mean(axis=1)
    preds_all_L = pd.concat(res_L, axis=1)
    for i , j, k in entry_combinations:
        preds_L[f"L_{i}_{j}_{k}"] = preds_all_L.filter(like = f"is_unique_equilibrium_{i}_{j}_{k}").mean(axis=1)
    merged = pd.concat((predicted_outcomes_mnl, preds_H, preds_L), axis=1)
    merged = merged.assign(market=range(len(merged)))
    merged_long = pd.wide_to_long(merged, stubnames = ["T", "H", "L"], i="market", j="equilibrium", sep = "_", suffix = '\w+')
    merged_long['deviation_minus'] = np.min(merged_long.T - merged_long.L,0).abs()**2 
    merged_long['deviation_plus'] = np.max(merged_long.T - merged_long.H,0).abs()**2
    collapsed = merged_long.groupby(['market'])[['deviation_plus', 'deviation_minus']].sum()
    total_norm = ((collapsed['deviation_plus'])**.5 + (collapsed['deviation_minus'])**.5).mean() 
    # form Euclidean norm
    return total_norm 
 

results_CT = minimize(
    get_mi,
    1,
    args=(mi_data, predicted_outcomes_mnl, shocks, params),
     tol=1e-6,
    method="Nelder-Mead",
    bounds=[(None, None)],
)

#%% 
#CT inference
min_CT = results_CT.fun
c0 = 1.25*min_CT

# Find initial confidence region by evaluating the obj function in a grid
# of 50 points (from -1 to 4.0) 
grid_input = np.arange(-1, 4.1, 0.1)
grid_output = np.array([get_mi(x, mi_data, predicted_outcomes_mnl, shocks, params) for x in grid_input])
# find the subregion where the objective function is less than c0 
refined_grid = grid_input[grid_output <= c0]
# get its bounds 
mu_low = refined_grid.min()
mu_high = refined_grid.max()

#do subsampling
def get_subsample(data, mi_data, predicted_outcomes_mnl, shocks, params):
    # get the original subsample 
    subsample_size = params["subsample_size"]
    sub_data = data.sample(subsample_size)
    markets_to_keep = sub_data.index.values
    predicted_outcomes_mnl_subsample  =predicted_outcomes_mnl[predicted_outcomes_mnl.index.isin(markets_to_keep)]
    mi_data_subsample = mi_data[mi_data['market'].isin(sub_data.index.values)]
    shocks_subsample = shocks[np.isin(shocks[:,0], markets_to_keep)]
    return mi_data_subsample, predicted_outcomes_mnl_subsample, shocks_subsample
#run on entire region to get test statistic 
def calc_mi_subsample(data, predicted_outcomes_mnl, shocks, params, grid):
    subsample_output = np.array([get_mi(x, data, predicted_outcomes_mnl, shocks, params) for x in grid])
    max_mi_sub = subsample_output.max()
    min_mi_sub = subsample_output.min()  
    return max_mi_sub - min_mi_sub 
# bootstrap wrapper 

BS_draws = 50
def bootstrap_wrapper(entry, mi_data, predicted_outcomes_mnl, shocks, params, refined_grid):
    mi_data_subsample, predicted_outcomes_mnl_subsample, shocks_subsample = get_subsample(entry, mi_data, predicted_outcomes_mnl, shocks, params)
    out = calc_mi_subsample(mi_data_subsample, predicted_outcomes_mnl_subsample, shocks_subsample, params, refined_grid)
    return out

results = Parallel(n_jobs=-1)(delayed(bootstrap_wrapper)(entry, mi_data, predicted_outcomes_mnl, shocks, params, refined_grid) for i in range(BS_draws)) 

# get quantile  
subsample_size = params["subsample_size"]
M = params["M"]
c1 = subsample_size/M * np.quantile(results, 0.95)
refined_grid = grid_input[grid_output -min_CT <= c1]
mu_low = refined_grid.min()
mu_high = refined_grid.max()

# %%
