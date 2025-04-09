# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
import main_helpers as mh

# %%
from importlib import reload

# %%
reload(mh)
# %%
PARAMS = {
    "T": 20_000,
    "mu": -1,
    "R": -3,
    "beta": 0.9,
    "gamma": 0.5775,  # Euler's constant
}
# %%
R = mh.RustSimulation(
    params=PARAMS,
    seed=273,
    verbose=True,
)
# %%
R.run_value_function_iteration()

# %%
R.run_data_simulation()
# %%
R.get_data_frequencies()
# %%
