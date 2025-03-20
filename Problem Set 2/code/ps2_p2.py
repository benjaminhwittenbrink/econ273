
#%%

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import toml

from scipy.optimize import minimize

#%%

with open("params_q2.toml", "r") as file:
    params = toml.load(file)
print(params)


#%%

np.random.seed(14_273)

def process_data(dir="../data/"):
    # 1. Load the data
    entry = pd.read_csv(os.path.join(dir, "entryData.csv"), header =None)
    entry.rename(columns={0: "X", 1: "Z_1", 2: "Z_2",  3: "Z_3", 4: "enter_1", 5: "enter_2", 6: "enter_3"}, errors="raise", inplace=True)
    entry = entry.assign(market =range(len(entry)))
    #reshape
    entry = pd.wide_to_long(entry, ['Z', 'enter'], i  = 'market', j = 'firm', sep = "_")
    #entry['join_index'] = range(0,len(entry))
    return entry
entry = process_data()

def make_shock_matrix(data, N_draws = params["N_draws"]):
    shocks =  np.random.normal(0, 1, size = (len(data), N_draws))
    #shocks = pd.DataFrame(shocks,  index = data.index).add_prefix("u_fm",1)
    #shocks['join_index'] = range(0,len(shocks))
    return shocks

shocks = make_shock_matrix(entry)

#test = pd.concat([entry, shocks], axis =1)
#%%

def get_simulated_prob_entry(params, data, alpha = params["alpha"], beta = params["beta"], delta = params["delta"] , N_draws = params["N_draws"]):
    mu = params[0]
    sigma = params[1]
    data['constant_profit_component'] = data.X*beta 
    for i in range( N_draws):
        data['phi_fm'] =  data.Z*alpha  + sigma*(mu + shocks[:,i])
        data = data.sort_values(by = ['market', 'phi_fm'], ascending = [True, True])
        data['firm_rank']= data.groupby('market').cumcount() + 1
        data['profits_if_enter'] = data.constant_profit_component  - delta*np.log(data.firm_rank) - data.phi_fm
        data['pos_prof'] = data.profits_if_enter > 0
        data['predict_entry{0}'.format(i)] = np.minimum(data.sort_values(by=['market','phi_fm'], ascending = [True,False]).groupby('market')['pos_prof'].cumsum(),1)
        data = data.drop(columns = ['phi_fm', 'firm_rank', 'profits_if_enter', 'pos_prof'])
    data['predicted_entry_mean'] = data.filter(like='predict_entry', axis = 1).mean(axis=1)
    data['llh'] = np.log(data['predicted_entry_mean']) * data['enter'] + np.log(1 - data['predicted_entry_mean']) * (1 - data['enter'])
    llh = - np.mean(data['llh'])
    return llh

results = minimize(
            get_simulated_prob_entry,
            [2,1],
            args=entry,
            tol=1e-14,
            method="Nelder-Mead",
            bounds=[(None, None),(.1, None)],
        )


#%%
def sim_likelihood(data, mu, sigma, N_draws = params["N_draws"], alpha = params["alpha"], beta = params["beta"]):
   #make spine 
    spine = data[['market', 'firm']]
    datasets = []
    for i in range(N_draws):
        datasets = datasets.append(i)
        i = spine.copy()
        "draw{0}".format(i)["predict{0}".format(i)] = get_simulated_prob_entry(data, mu, sigma, alpha, beta)
    return data, datasets

entry = sim_likelihood(entry, mu= 0, sigma =1)

# %%