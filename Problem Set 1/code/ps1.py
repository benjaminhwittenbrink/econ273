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
# generate simulated data given params (returns X, p, s, W, Z)
data = simulate.DemandData(params, seed=14_273)

# %%
s, p = data.simulate()
