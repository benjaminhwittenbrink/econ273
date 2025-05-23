# %%
import time
import logging
import toml

import pandas as pd
import numpy as np

from tqdm import tqdm
from importlib import reload

import estimate
import data
import utils

# %%

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

reload(data)
reload(estimate)

# %%
with open("params.toml", "r") as file:
    params = toml.load(file)


def convert_to_latex_macros(dictionary, prefix=""):
    macros = []
    for key, value in dictionary.items():
        latex_key = f"{prefix}{key}"
        latex_key = "params" + latex_key.replace("_", "")
        if isinstance(value, dict):  # Handle nested sections
            macros.extend(convert_to_latex_macros(value, prefix=f"{latex_key}_"))
        elif isinstance(value, list):  # Handle lists
            value_str = ", ".join(map(str, value))
            macros.append(f"\\newcommand{{\\{latex_key}}}{{\\{{{value_str}\\}}}}")
        else:  # Handle scalar values
            macros.append(f"\\newcommand{{\\{latex_key}}}{{{value}}}")
    return macros


# Generate LaTeX macros
latex_macros = convert_to_latex_macros(params)

utils.write_params(params, file_path="../variables.tex", prefix="params")
print("variables.tex has been generated!")
# %%
default_seed = 14_273
N_iters = 1

dfs = []
for s in tqdm([default_seed + i for i in range(N_iters)]):
    logging.info(f"Simulating with seed {s}.")
    try:
        start = time.time()
        DD = data.DiamondData(params, seed=s)
        DD.simulate()
        DD.write()
        df = DD.to_dataframe()
        dfs.append(df)
        end = time.time()
        # logging.info(
        #     f"Simulation with seed {s} completed in {end - start:.2f} seconds."
        # )
    except Exception as e:
        logging.error(f"Error with seed {s}: {e}")
        continue

if len(dfs) == 0:
    raise ValueError("No successful simulations. Check the logs for errors.")

df = pd.concat(dfs, ignore_index=True)

# %% Plot descriptive stats
utils.plot_descriptive_stats(df)

# %%
reload(estimate)
DM = estimate.DiamondModel(df, DD, seed=default_seed, verbose=True)
DM.initialize()

# %%
DM.fit(theta0=np.array([4, 1]))
DM.print_results()
utils.write_params(DM.est_params, file_path="../estimates.tex", prefix="paramsEst")
# %%
# DM.run_regulation_counterfactual()
output_folder = "/Users/vbp/Princeton Dropbox/Veronica Backer Peral/Apps/Overleaf/econ273/Mini_Project/figures"
DM.run_amenity_counterfactual(output_folder)

# %%
