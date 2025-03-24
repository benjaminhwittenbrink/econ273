# %%
import data
import os
import logging
import toml
from importlib import reload

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

DD = data.DiamondData(params)
DD.simulate()
# %%
