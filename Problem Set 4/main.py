# %%
import numpy as np


# %%
import main_helpers as mh

# %%

from importlib import reload

reload(mh)

# %%
PARAMS = {"A": 0.3, "B": 0.6, "C": 0.15, "delta": 0.8}

# %% Solve for equilibrium
EE = mh.EntryExit(PARAMS, verbose=True)
EE.solve_system()

# %% Generate data
states, psi, phi = EE.simulate_data()

print(states[:, 0])
print(states[:, 1])

print("Psi:")
print(psi[:, 0].round(2))
print(psi[:, 1].round(2))

print("Phi:")
print(phi[:, 0].round(2))
print(phi[:, 1].round(2))

# %% Solve for parameters
