"""
Title: Problem Set 1
Author: Benjamin Wittenbrink, Jack Kelly, Veronica Backer Peral
Date: 03/01/25
"""

# %%
import toml
import logging
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.optimize as opt

# %%
from importlib import reload

# %%
import simulate
import blp

# %%
reload(simulate)
reload(blp)
# %%
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
# %%
# Load parameters from params.toml
with open("params.toml", "r") as file:
    params = toml.load(file)
print(params)

# %%
# generate simulated data given params
data = simulate.DemandData(params, seed=14_273, verbose=True)

# %%
s, p, delta = data.simulate()

# %%
res = data.compute_empirical_moments()
# %%
# res_l = []
# rs = np.random.randint(0, 1_000_000_000)
# for b in range(1000):
#     if b % 100 == 0:
#         print(f"Iteration {b}")
#     data = simulate.DemandData(params, seed=rs + b, verbose=False)
#     s, p = data.simulate()
#     res = data.compute_empirical_moments()
#     res_l.append(res)

# %%
reload(blp)
# %%
blp_est = blp.BLP(data, tol=1e-14, verbose=True)
# %%
alpha_hat, beta_hat, sigma_alpha_hat = blp_est.run_gmm_2stage()
# %%
print(f"alpha_hat: {np.round(alpha_hat, 5)} vs. {params['alpha']}")
print(f"beta_hat: {np.round(beta_hat, 5)} vs. {params['betas']}")
print(f"sigma_alpha_hat: {np.round(sigma_alpha_hat, 5)} vs. {params['sigma_alpha']}")

# %%
