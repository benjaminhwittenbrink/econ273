"""
Title: Problem Set 1 -- blp.py
Author: Benjamin Wittenbrink, Jack Kelly, Veronica Backer Peral
Date: 03/01/25
"""


class BLP:

    def __init__(self, data, params):
        self.data = data
        self.params = params

    def _compute_gmm_obj(self, **kwargs):
        # helper fns: _invert_shares, _compute_moment_conditions, _construct_instruments
        pass

    def _invert_shares(self, **kwargs):
        # invert shares by running delta contraction mapping
        pass

    def _compute_moment_conditions(self, **kwargs):
        pass

    def _construct_instruments(self, **kwargs):
        pass

    def run_demand_estimation(self, **kwargs):
        # simulate Pns - hold fixed for theta

        # for each theta

        # compute delta i.e., invert shares

        # solve for xi, omega

        # compute moment condition G
        pass
