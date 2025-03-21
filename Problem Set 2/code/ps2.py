# %%
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

# %%
from importlib import reload

# %%
import ps2_p1 as p1

# %%
reload(p1)
# %%
# Part 1
gmd = p1.load_data()
# %%
regs = p1.replicate_GM(gmd)

# %%
table = p1.output_GM_table(regs)

# %%
print(table.to_latex())

# %%
# Part 2
acf = p1.ACF(df=gmd)
# acf_rho_diff_res = acf.est_rho_diff_model()
# acf_rho_diff_res
# %%
first_stage = acf.est_first_stage(degree=3)
results = acf.est_second_stage()
print(np.round(results, 4))
# %%
