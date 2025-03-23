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
#set seed
np.random.seed(14_273)
#read in data
def process_data(dir="../data/"):
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
# make a long version of the data for simulation
def make_long(data):
    data = data.assign(market=range(len(data)))
    # reshape
    data_long = pd.wide_to_long(data, ["Z", "enter"], i="market", j="firm", sep="_")
    data_long = data_long.reset_index().sort_values(["market", "firm"])
    return data_long
# make a matrix of shocks for the simulations. Only draw once and rescale by mean and variance later to avoid chattering.
def make_shock_matrix(data, N_draws=params["N_draws"], F = params["F"]):
    shocks = np.random.normal(0, 1, size=(len(data), N_draws))
    market = np.floor(np.array(range(len(shocks)))/F)
    shocks = np.column_stack((market, shocks))
    return shocks

# form datasets
entry = process_data()
entry_long = make_long(entry)
shocks = make_shock_matrix(entry_long)


# %%
##### BERRY #####
#function that gets simulated log likelihood
def get_simulated_prob_entry(theta, data, shocks, params, fixed_sigma=False):
    mu = theta[0]
    # allow for fixed sigma to make CT comparison more natural
    if fixed_sigma == True: sigma = params["sigma"]
    else: sigma = theta[1]
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

# maximize log likelihood
results_Berry = minimize(
    get_simulated_prob_entry,
    [1, .5],
    args=(entry_long, shocks, params),
     tol=1e-6,
    method="Nelder-Mead",
    bounds=[(None, None), (0.1, None)],
)

# do version with fixed sigma
results_Berry_fixed_sigma = minimize(
    get_simulated_prob_entry,
    1,
    args=( entry_long, shocks, params, True),
     tol=1e-6,
    method="Nelder-Mead",
    bounds=[(None, None)],
)



# %%
##### CILIBERTO AND TAMER ###### 
# function that forms multinomial logit predictions
def get_mnl(data):
    data['entry_vector'] = data.enter_1.astype(str) + '_' + data.enter_2.astype(str) + '_' + data.enter_3.astype(str)
    regressors = pd.DataFrame(np.column_stack(( np.ones(len(entry)), entry.X , entry.Z_1, entry.Z_2,  entry.Z_3)))
    mnl = LogisticRegression(multi_class='multinomial').fit(regressors, entry.entry_vector)
    #get predictions
    predicted_outcomes_mnl = pd.DataFrame(mnl.predict_proba(regressors))
    #rename predicted to be the outomces
    predicted_outcomes_mnl.columns = mnl.classes_.tolist()
    predicted_outcomes_mnl.columns = 'T_' + predicted_outcomes_mnl.columns
    # fill in the remaining columns that are never observed with 0 
    for i in range (2): 
        for j in range(2): 
            for k in range(2): 
                if f'T_{i}_{j}_{k}' not in predicted_outcomes_mnl.columns: 
                    predicted_outcomes_mnl[f'T_{i}_{j}_{k}'] = 0
    # make sure the order aligns with the one we will use for simulation
    predicted_outcomes_mnl = np.array(predicted_outcomes_mnl[sorted(predicted_outcomes_mnl.columns)])
    market = np.array(range(len(predicted_outcomes_mnl)))
    predicted_outcomes_mnl = np.column_stack((market, predicted_outcomes_mnl))
    return predicted_outcomes_mnl
# helper function that prepares the data for the minimization routine
def prep_for_mi(data, params): 
    beta = params["beta"]
    mi_data = data.copy()
    mi_data["constant_profit_component"] = mi_data["X"] * beta 
    mi_data["constant_phi_component"] = data["Z"] * alpha
    # store the actual equilibrium in the data 
    mi_data = mi_data.sort_values(by=["market", "firm"], ascending=[True, True])
    return mi_data

# function that gets the equilibria for a given draw 
def get_equilibria(n, mu, data, shocks, params):
        alpha, beta, delta, N_draws, sigma = (
            params["alpha"],
            params["beta"],
            params["delta"],
            params["N_draws"],
            params["sigma"],
        )
        tmp_data = data.copy()
        entry_combinations = [(i, j, k) for i in range(2) for j in range(2) for k in range(2)]
        # calculate firm specific phi
        phi_fm =  tmp_data["constant_phi_component"] + sigma * (mu + shocks[:, n+1])
        tmp_data = tmp_data.filter(["market", "firm","constant_profit_component"])
        # estimate profits from each possible entry and vector and determine if it's an equilibrium
        for i , j, k in entry_combinations:
            # get profits if this many firms enter 
            if i+j+k>0:   
                profits_in_equilibrium = (i*(tmp_data['firm']==1)+ j*(tmp_data['firm']==2) + k*(tmp_data['firm']==3)) *(
                    tmp_data["constant_profit_component"]
                    - delta * np.log(i+j+k)
                    - phi_fm)
                tmp_data[f"profits_if_enter_{i}_{j}_{k}"] = profits_in_equilibrium
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
            tmp_data["check_equilibrium"] = (tmp_data.groupby(["market"])["profitable_deviation"].max() == 0)*1
            tmp_data["check_equilibrium"] = tmp_data.groupby(["market"])["check_equilibrium"].ffill()
            tmp_data = tmp_data.drop(columns = "profitable_deviation")
            tmp_data = tmp_data.rename(columns={"check_equilibrium":f"is_equilibrium_{i}_{j}_{k}_draw_{n}"})
        # check for unique equilibria 
        tmp_data = tmp_data[tmp_data['firm']==1]
        tmp_data = tmp_data.drop(columns = list(tmp_data.filter(regex = "profits_if_enter_")))
        tmp_data = tmp_data.drop(columns = ["constant_profit_component", "market", "firm"]) 
        unique_equilibrium = tmp_data.filter(like="is_equilibrium_").sum(axis=1) == 1
        tmp_data_unique =tmp_data.mul(unique_equilibrium, axis=0)
        all = np.array([tmp_data, tmp_data_unique])
        return all  
   
# wrapper over simulation draws to get MI objective function   
def get_mi(mu, data, predicted_outcomes_mnl, shocks, params):   
    #   calculate equilibria over all simulation draws
        out= np.array([get_equilibria(n, mu, data, shocks, params) for n in range(N_draws)])
        out_H = out[:,0,:,:]
        out_L = out[:,1,:,:]
         # combine all entry predictions
        preds_all_H = np.mean(out_H , axis=0)
        preds_all_L = np.mean(out_L , axis=0)
        zeroes = np.zeros(M)
        predicted_outcomes_mnl_tmp = predicted_outcomes_mnl[:,1:]
        deviation_minus  = ((predicted_outcomes_mnl_tmp - preds_all_L)*(predicted_outcomes_mnl_tmp - preds_all_L<0))**2
        deviation_plus  =  ((predicted_outcomes_mnl_tmp - preds_all_H)*(predicted_outcomes_mnl_tmp - preds_all_L>0))**2
        # take euclidean norm across equilibria 
        deviation_minus =  np.sqrt(np.sum(deviation_minus, axis=1))
        deviation_plus =  np.sqrt(np.sum(deviation_minus, axis=0))
        # mean over all markets 
        deviation_minus = np.mean(deviation_minus)
        deviation_plus = np.mean(deviation_plus)
        return deviation_minus +  deviation_plus

# form datasets and run minimization routine
predicted_outcomes_mnl = get_mnl(entry)
mi_data = prep_for_mi(entry_long, params)

results_CT = minimize(
    get_mi,
    1,
    args=(mi_data, predicted_outcomes_mnl, shocks, params),
     tol=1e-6,
    method="Nelder-Mead",
    bounds=[(None, None)],
)

#%% 
#####CT inference#####
 # function that does the subsampling 
def get_subsample(data, mi_data, predicted_outcomes_mnl, shocks, params):
    # get the original subsample 
    subsample_size = params["subsample_size"]
    # subset all the relevant datasets/arrays corresponding to these markets
    sub_data = data.sample(subsample_size)
    markets_to_keep = sub_data.index.values
    predicted_outcomes_mnl_subsample  =predicted_outcomes_mnl[np.isin(predicted_outcomes_mnl[:,0], markets_to_keep)]
    mi_data_subsample = mi_data[mi_data['market'].isin(sub_data.index.values)]
    shocks_subsample = shocks[np.isin(shocks[:,0], markets_to_keep)]
    return mi_data_subsample, predicted_outcomes_mnl_subsample, shocks_subsample

# function that calculates objective function on every point in the grid for a given subsample draw
def calc_mi_subsample(data, predicted_outcomes_mnl, shocks, params, grid):
    subsample_output = np.array([get_mi(x, data, predicted_outcomes_mnl, shocks, params) for x in grid])
    max_mi_sub = subsample_output.max()
    min_mi_sub = subsample_output.min()  
    return max_mi_sub - min_mi_sub 

# bootstrap wrapper
def bootstrap_wrapper(data, mi_data, predicted_outcomes_mnl, shocks, params, refined_grid):
    mi_data_subsample, predicted_outcomes_mnl_subsample, shocks_subsample = get_subsample(data, mi_data, predicted_outcomes_mnl, shocks, params)
    out = calc_mi_subsample(mi_data_subsample, predicted_outcomes_mnl_subsample, shocks_subsample, params, refined_grid)
    return out

# function that finds confidence region given c
def get_confidence_region(c, search_grid, data, predicted_outcomes_mnl, shocks, params):
    grid_output = np.array([get_mi(x, mi_data, predicted_outcomes_mnl, shocks,params) for x in search_grid])
    # find the subregion where the objective function is less than c0 
    refined_grid = search_grid[grid_output <= c0]
    # get its bounds 
    mu_low = refined_grid.min()
    mu_high = refined_grid.max()
    return refined_grid, mu_low, mu_high, grid_output

# function that gets refined quantile and confidence set 
def get_quantile_and_CI(grid_input, grid_output, min_CT, results, params, CI_size):
    subsample_size = params["subsample_size"]
    M =  params["M"]
    # get the quantile of the subsample output
    c = subsample_size/M * np.quantile(results, CI_size)
    refined_grid = grid_input[grid_output -min_CT <= c1]
    mu_low = refined_grid.min()
    mu_high = refined_grid.max()
    return c, mu_low, mu_high
#overall inference wrapper
def CT_inference(data, mi_data, predicted_outcomes_mnl, shocks, params, min_CT, c0, CI_size):
    BS_draws = params["BS_draws"]
    # get initial CR using c0 
    search_grid = np.arange(-4, 4.1, .1)
    refined_grid, mu_low, mu_high, grid_output = get_confidence_region(c0, search_grid, mi_data, predicted_outcomes_mnl, shocks, params)
    # apply subsampling procedure to each element of the refined_grid 
    bs_draws = Parallel(n_jobs=-1)(delayed(bootstrap_wrapper)(entry, mi_data, predicted_outcomes_mnl, shocks, params, refined_grid) for i in range(BS_draws)) 
    c, mu_low, mu_high = get_quantile_and_CI(search_grid, grid_output, min_CT, bs_draws, params, CI_size)
    # iterate : this time, just refine the grid with the new c, don't use finer grid yet 
    refined_grid = search_grid[grid_output - min_CT <= c]
    bs_draws = Parallel(n_jobs=-1)(delayed(bootstrap_wrapper)(entry, mi_data, predicted_outcomes_mnl, shocks, params, refined_grid) for i in range(BS_draws)) 
    c, mu_low, mu_high = get_quantile_and_CI(search_grid, grid_output, min_CT, bs_draws, params, CI_size)
    # final iteration: this time, we use a finer grid
    search_grid = np.arange(-4, 4.1, .05)
    refined_grid, mu_low, mu_high, grid_output = get_confidence_region(c, search_grid, mi_data, predicted_outcomes_mnl, shocks, params)
    bs_draws = Parallel(n_jobs=-1)(delayed(bootstrap_wrapper)(entry, mi_data, predicted_outcomes_mnl, shocks, params, refined_grid) for i in range(BS_draws)) 
    c, mu_low, mu_high = get_quantile_and_CI(search_grid, grid_output, min_CT, bs_draws, params, CI_size)
    return  mu_low, mu_high

# get the minimum of the CT objective for initial guess and min_CT
min_CT = results_CT.fun
c0 = 1.25*min_CT
# run the wrapper
CI_L, CI_H = CT_inference(entry, mi_data, predicted_outcomes_mnl, shocks, params, min_CT, c0, .95)

# %%
### make latex table of results
headers = ["Berry","Berry (fixed $\sigma^2$ =1)","Ciliberto and Tamer (95\% CI)"]
data = dict()
data["$\hat{\mu}$"] = [ round(results_Berry.x[0],3), round(results_Berry_fixed_sigma.x[0],3) , '[' + str(round(CI_L, 3)) + ', ' + str(round(CI_H, 3)) + ']']
data["$\hat{\sigma^2}$"] = [round(results_Berry.x[1],3), None , None ]
textabular = f"l|{'r'*len(headers)}"
texheader = " & " + " & ".join(headers) + "\\\\"
texdata = "\\hline\n"
for label in sorted(data):
   if label == "z":
      texdata += "\\hline\n"
   texdata += f"{label} & {' & '.join(map(str,data[label]))} \\\\\n"

out_string = "\\begin{tabular}{"+textabular+"}" + texheader + texdata + "\\end{tabular}"
f = open("../tables/entry_table.tex", "w")
f.write(out_string)
f.close()
# %%
