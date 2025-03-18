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
regs = p1.replicate_GM(gmd)

# %%
table = p1.output_GM_table(regs)

# %%
print(table.to_latex())

# %%
# Part 2
# part 1
acf_rho_res = p1.acf_rho_diff(gmd)

# %%
acf_rho_res
# %%
