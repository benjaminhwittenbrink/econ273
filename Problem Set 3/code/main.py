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
PARAMS = {"T": 20_000, "mu": -1, "R": -3, "beta": 0.9, "N_sim": 1_000}
# %%
R = mh.MachineReplacementData(
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
mod = mh.MachineReplacementEstimation(
    data=R.get_data(),
    params={"beta": 0.9, "T": PARAMS["T"]},
    verbose=True,
)


# % %
reload(mh)
mod = mh.MachineReplacementEstimation(
    data=R.get_data(),
    params={"beta": 0.9, "T": PARAMS["T"]},
    verbose=True,
)
mod.estimate_theta(
    approach="Forward Simulation",
    N_sim=PARAMS["N_sim"],
    # T=PARAMS["T"],
    # F0=F0,
    # F1=F1,
)
mod.get_theta_hat()
# %%
mod.estimate_theta(approach="Rust")
Rust = mod.get_theta_hat()


# %%
mod.estimate_theta(
    approach="Forward Simulation",
    N_sim=PARAMS["N_sim"],
    # T=PARAMS["T"],
    # F0=F0,
    # F1=F1,
)
# %%
F0 = np.array(
    [
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1],
    ]
)
F1 = np.array(
    [
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
    ]
)
mod.estimate_theta(
    approach="Forward Simulation",
    N_sim=PARAMS["N_sim"],
    # T=PARAMS["T"],
    # F0=F0,
    # F1=F1,
)
ForSim = mod.get_theta()


# %%
mod.estimate_theta(approach="Analytical")
AM = mod.get_theta()


# %%
# make latex table of results
### make latex table of results
headers = ["Nested Fixed Point", "Arcidiacono-Miller", "Forward Simulation"]
data = dict()
data["$\hat{\mu}$"] = [round(Rust[0], 3), round(AM[0], 3), round(ForSim[0], 3)]
data["$\hat{R}$"] = [round(Rust[1], 3), round(AM[1], 3), round(ForSim[1], 3)]
textabular = f"l|{'r'*len(headers)}"
texheader = " & " + " & ".join(headers) + "\\\\"
texdata = "\\hline\n"
for label in sorted(data):
    if label == "z":
        texdata += "\\hline\n"
    texdata += f"{label} & {' & '.join(map(str,data[label]))} \\\\\n"


out_string = (
    "\\begin{tabular}{" + textabular + "}" + texheader + texdata + "\\end{tabular}"
)
f = open("../tables/results.tex", "w")
f.write(out_string)
f.close()


# %%
