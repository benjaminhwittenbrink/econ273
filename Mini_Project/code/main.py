# %%
import data
import os
import logging
import toml
from importlib import reload
import pickle

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

logging.info("Logger is configured and working.")

# %%
reload(data)

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

# Save to a LaTeX file
with open("../variables.tex", "w") as f:
    for key, value in params.items():
        key = "params_" + key
        if isinstance(value, str):
            f.write(f"\\newcommand{{\\{key}}}{{{value}}}\n")
        else:
            f.write(f"\\newcommand{{\\{key}}}{{{value}}}\n")

print("variables.tex has been generated!")
DD = data.DiamondData(params)
DD.simulate()
# %%
DD.print_results()
# %%
res = DD.to_dataframe()
# %%
res.to_csv("../data/simulated_data_05_05.csv", index=False)
# %%
with open("../data/simulated_data_05_05.pkl", "wb") as f:
    pickle.dump(res, f)
# %%
