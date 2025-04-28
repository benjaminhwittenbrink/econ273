# %%
import numpy as np


# %%
import main_helpers as mh

# %%

from importlib import reload

reload(mh)

# %%
PARAMS = {"A": 0.3, "B": 0.6, "C": 0.15, "delta": 0.8}

# %%
EE = mh.EntryExit(PARAMS, verbose=True)

# %%
EE.solve_system()

# %%
