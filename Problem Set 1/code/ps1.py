"""
Title: Problem Set 1
Author: Benjamin Wittenbrink, Jack Kelly, Veronica Backer Peral
Date: 03/01/25
"""

# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import toml

import scipy.optimize as opt

# %%
from importlib import reload

# %%
import simulate

# %%
reload(simulate)
# %%
# Load parameters from params.toml
with open("params.toml", "r") as file:
    params = toml.load(file)
print(params)

# %%
# set seed
np.random.seed(14_273)

# %%
# generate simulated data given params
data = simulate.DemandData(params, seed=14_273, verbose=True)

# %%
s, p, delta = data.simulate()

# %%

# %%
res = data.compute_empirical_moments()
# %%
res_l = []
rs = np.random.randint(0, 1_000_000_000)
for b in range(1000):
    if b % 100 == 0:
        print(f"Iteration {b}")
    data = simulate.DemandData(params, seed=rs + b, verbose=False)
    s, p = data.simulate()
    res = data.compute_empirical_moments()
    res_l.append(res)

# %%
