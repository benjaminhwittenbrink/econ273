import numpy as np
from scipy import optimize as opt


class EntryExit:
    def __init__(self, params, verbose=False):
        self.params = params
        self.verbose = verbose

        self._result_order = ["V00", "V01", "V10", "V11", "p00", "p01", "p10", "p11"]

    def set_params(self, params):
        self.params = params

    def set_verbose(self, verbose):
        self.verbose = verbose

    def get_params(self):
        return self.params

    def print_results(self):
        """
        Print the results of the system of equations.
        """
        if self.verbose:
            print("Parameters:", self.params)
            print("Results:")
            for key, value in self.results.items():
                print(f"{key}: {value}")

    def solve_system(self, initial_guess=None):
        """
        Solve the system of equations using a root-finding algorithm.
        """
        if initial_guess is None:
            initial_guess = np.ones(8)

        sol = opt.root(
            self._optimize_system_func,
            initial_guess,
            method="hybr",
            options={"xtol": 1e-8},
        )
        if sol.success:
            if self.verbose:
                print("Solution found:", sol.x)
            self.results = dict(zip(self._result_order, sol.x))
        else:
            raise ValueError("Root finding failed: " + sol.message)

    def _optimize_system_func(self, x):
        """
        Objective function to be minimized.
        """
        # Unpack the parameters
        V00, V01, V10, V11, p00, p01, p10, p11 = x
        A, B, C, delta = (
            self.params["A"],
            self.params["B"],
            self.params["C"],
            self.params["delta"],
        )

        # helpful defs
        S11_01 = V11 + V01
        S10_00 = V10 + V00
        D11_01 = V11 - V01
        D10_00 = V10 - V00

        # Construct the system of equations
        # value function equations
        # 1. V(0, 0)
        V00_frac_term = (C + delta * (p00 * S11_01 + (1 - p00) * S10_00)) / 2
        V00_rhs = p00 * V00_frac_term + (1 - p00) * (
            delta * (p00 * V01 + (1 - p00) * V00)
        )

        # 2. V(0, 1)
        V01_frac_term = (-C + delta * (p10 * S11_01 + (1 - p10) * S10_00)) / 2
        V01_rhs = p01 * V01_frac_term + (1 - p01) * (
            delta * (p10 * V01 + (1 - p10) * V00)
        )

        # 3. V(1, 1)
        V11_first_term = 2 * A - B + delta * (p11 * V11 + (1 - p11) * V10)
        V11_frac_term = (
            4 * A - 2 * B + 1 + delta * (p11 * S11_01 + (1 - p11) * S10_00)
        ) / 2
        V11_rhs = p11 * V11_first_term + (1 - p11) * V11_frac_term

        # 4. V(1, 0)
        V10_first_term = 2 * A + delta * (p01 * V11 + (1 - p01) * V10)
        V10_frac_term = (4 * A + 1 + delta * (p01 * S11_01 + (1 - p01) * S10_00)) / 2
        V10_rhs = p10 * V10_first_term + (1 - p10) * V10_frac_term

        # probability equations
        # 5. p(0, 0)
        p00_raw = delta * (p00 * D11_01 + (1 - p00) * D10_00) - C
        p00_rhs = np.clip(p00_raw, 0, 1)

        # 6. p(0, 1)
        p01_raw = delta * (p01 * D11_01 + (1 - p01) * D10_00) - C
        p01_rhs = np.clip(p01_raw, 0, 1)

        # 7. p(1, 1)
        p11_raw = delta * (p11 * D11_01 + (1 - p11) * D10_00)
        p11_rhs = np.clip(p11_raw, 0, 1)

        # 8. p(1, 0)
        p10_raw = delta * (p10 * D11_01 + (1 - p10) * D10_00)
        p10_rhs = np.clip(p10_raw, 0, 1)

        # return the system of equations
        return np.array(
            [
                V00 - V00_rhs,
                V01 - V01_rhs,
                V10 - V10_rhs,
                V11 - V11_rhs,
                p00 - p00_rhs,
                p01 - p01_rhs,
                p10 - p10_rhs,
                p11 - p11_rhs,
            ]
        )
