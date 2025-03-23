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

# %%
acf_rho_diff_res = acf.est_rho_diff_model()
acf_rho_diff_res
# %%
first_stage = acf.est_first_stage(degree=3)

# %%
rho, mu, beta1, beta2, beta3 = acf.est_second_stage()
print(f"rho: {rho:.3f}")
print(f"beta1: {beta1:.3f}")
print(f"beta2: {beta2:.3f}")
print(f"beta3: {beta3:.3f}")

labor_elas = beta1

# %%
rho, beta1, beta2, beta3 = acf.est_second_stage_alt()
print(f"rho: {rho:.3f}")
print(f"beta1: {beta1:.3f}")
print(f"beta2: {beta2:.3f}")
print(f"beta3: {beta3:.3f}")

# %%
rho, beta1, beta2, beta3, alpha = acf.est_second_stage_survival_control()
print(f"rho: {rho:.3f}")
print(f"beta1: {beta1:.3f}")
print(f"beta2: {beta2:.3f}")
print(f"beta3: {beta3:.3f}")
print(f"alpha: {alpha:.3f}")

# %% Estimate markups
p1.estimate_markups(labor_elas)
# %%
