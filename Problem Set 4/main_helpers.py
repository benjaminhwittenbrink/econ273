import numpy as np
from scipy import optimize as opt
from scipy.optimize import minimize


class EntryExit:
    def __init__(self, params, verbose=False, seed=14_273):
        self.params = params
        self.verbose = verbose
        self.seed = seed
        np.random.seed(seed)

        self.states = None

        self._result_order = ["V00", "V01", "V10", "V11", "p00", "p01", "p10", "p11"]

    def set_params(self, params):
        self.params = params

    def set_verbose(self, verbose):
        self.verbose = verbose

    def get_params(self):
        return self.params

    def psi_draw(self):
        return np.random.uniform(0, 1)

    def phi_draw(self):
        return np.random.uniform(0, 1)

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
            initial_guess = np.array([0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4])

        sol = opt.root(
            self._optimize_system_func,
            initial_guess,
            method="hybr",
            options={"xtol": 1e-8},
            args=(
                self.params["A"],
                self.params["B"],
                self.params["C"],
            ),
        )
        if sol.success:
            self.results = dict(zip(self._result_order, sol.x))
            if self.verbose:
                print("Solution found:")
                for key, value in self.results.items():
                    # Print rounded result to 2 decimal places
                    print(f"{key}: {value:.2f}")
        else:
            raise ValueError("Root finding failed: " + sol.message)

    def _optimize_system_func(self, x, A, B, C):
        """
        Objective function to be minimized.
        """
        # Unpack the parameters
        V00, V01, V10, V11, p00, p01, p10, p11 = x

        # helpful defs
        q00 = 1 - p00
        q01 = 1 - p01
        q10 = 1 - p10
        q11 = 1 - p11

        delta = self.params["delta"]
        pi10 = 2 * A
        pi11 = 2 * A - B

        #### Construct the system of equations
        # value‐function updates
        V00_new = -p00 * (C + p00 / 2) + delta * (
            p00 * (p00 * V11 + q00 * V10) + q00 * (p00 * V01 + q00 * V00)
        )

        V01_new = -p10 * (C + p10 / 2) + delta * (
            p10 * (p01 * V11 + q01 * V10) + q10 * (p01 * V01 + q01 * V00)
        )

        V10_new = (
            pi10
            + (q01 * (1 + p01)) / 2
            + delta * (p01 * (p10 * V11 + q10 * V10) + q01 * (p10 * V01 + q10 * V00))
        )

        V11_new = (
            pi11
            + (q11 * (1 + p11)) / 2
            + delta * (p11 * (p11 * V11 + q11 * V10) + q11 * (p11 * V01 + q11 * V00))
        )

        # probability function updates
        p00_new = -C + delta * (p00 * (V11 - V01) + q00 * (V10 - V00))
        p10_new = -C + delta * (p01 * (V11 - V01) + q01 * (V10 - V00))
        p01_new = delta * (p10 * (V11 - V01) + q10 * (V10 - V00))
        p11_new = delta * (p11 * (V11 - V01) + q11 * (V10 - V00))

        # p00_new = np.clip(p00_new, 0, 1)
        # p01_new = np.clip(p01_new, 0, 1)
        # p10_new = np.clip(p10_new, 0, 1)
        # p11_new = np.clip(p11_new, 0, 1)

        res = np.array(
            [
                V00 - V00_new,
                V01 - V01_new,
                V10 - V10_new,
                V11 - V11_new,
                p00 - p00_new,
                p01 - p01_new,
                p10 - p10_new,
                p11 - p11_new,
            ]
        )

        return res

    def _val_enter_exit(self, psi, phi, my_state, other_state):
        """
        Calculate the value of entering or exiting the market.
        """

        # Probability that the other player enters, given the current state
        p = self.results[f"p{my_state}{other_state}"]

        EV_in = p * self.results["V11"] + (1 - p) * self.results["V10"]
        EV_out = p * self.results["V01"] + (1 - p) * self.results["V00"]

        if my_state == 0:
            profit = 0
            enter_cost = self.params["C"] + psi
            scraps = 0
        elif my_state == 1 and other_state == 0:
            profit = 2 * self.params["A"]
            enter_cost = 0
            scraps = phi
        elif my_state == 1 and other_state == 1:
            profit = 2 * self.params["A"] - self.params["B"]
            enter_cost = 0
            scraps = phi

        val_enter = profit - enter_cost + self.params["delta"] * EV_in
        val_exit = profit + scraps + self.params["delta"] * EV_out

        return val_enter, val_exit

    def _choose_next_state(self, my_state, other_state):
        psi = self.psi_draw()
        phi = self.phi_draw()

        val_enter, val_exit = self._val_enter_exit(psi, phi, my_state, other_state)
        next_state = int(np.argmax([val_exit, val_enter]))
        return next_state, psi, phi

    def simulate_data(self, num_periods=1000):
        """
        Simulate data based on the solved system of equations.
        """

        states = []  # Initialize at (0,0)
        psi_draws = []
        phi_draws = []
        for t in range(num_periods):
            if t == 0:
                state = (0, 0)
            else:
                state = states[-1]

            next_state1, psi1, phi1 = self._choose_next_state(state[0], state[1])
            next_state2, psi2, phi2 = self._choose_next_state(state[1], state[0])

            states.append((next_state1, next_state2))
            psi_draws.append((psi1, psi2))
            phi_draws.append((phi1, phi2))

        self.states = states
        self.psi_draws = psi_draws
        self.phi_draws = phi_draws

    def _value_function(self, params, choice_probs):
        """
        Calculate the value function based on the parameters and choice probabilities.
        """
        A, B, C = params
        p00, p01, p10, p11 = choice_probs

    def _estimate_probs(self, params):
        A, B, C = params

        # Estimate value function and probabilities
        sol = opt.root(
            self._optimize_system_func,
            np.ones(8),
            method="hybr",
            options={"xtol": 1e-8},
            args=(
                A,
                B,
                C,
            ),
        )
        probs_est = sol.x[4:]

        return probs_est

    def _objective_function(self, params, probs_true):
        """
        Objective function to be minimized.
        """
        probs_est = self._estimate_probs(params)
        diff = probs_est - probs_true
        return np.abs(diff)

    def estimate_model(self):
        """
        Estimate the model parameters based on the simulated data.
        """
        if self.states is None:
            self.simulate_data()

        # Calculate probabilities
        p00_list = []
        p01_list = []
        p10_list = []
        p11_list = []
        for i, state in enumerate(self.states):
            if i == 0:
                continue

            last_state = self.states[i - 1]
            if last_state == (0, 0):
                if state[0] == 1:
                    p00_list.append(1)
                else:
                    p00_list.append(0)

                if state[1] == 1:
                    p00_list.append(1)
                else:
                    p00_list.append(0)

            elif last_state == (0, 1):
                if state[0] == 1:
                    p10_list.append(1)
                else:
                    p10_list.append(0)

                if state[1] == 1:
                    p01_list.append(1)
                else:
                    p01_list.append(0)

            elif last_state == (1, 0):
                if state[0] == 1:
                    p01_list.append(1)
                else:
                    p01_list.append(0)

                if state[1] == 1:
                    p10_list.append(1)
                else:
                    p10_list.append(0)

            elif last_state == (1, 1):
                if state[0] == 1:
                    p11_list.append(1)
                else:
                    p11_list.append(0)

                if state[1] == 1:
                    p11_list.append(1)
                else:
                    p11_list.append(0)

        p00 = np.mean(p00_list)
        p01 = np.mean(p01_list)
        p10 = np.mean(p10_list)
        p11 = np.mean(p11_list)
        probs_empirical = np.array([p00, p01, p10, p11])
        probs_true = np.array(
            [
                self.results["p00"],
                self.results["p01"],
                self.results["p10"],
                self.results["p11"],
            ]
        )

        initial_guess = np.array([0.5, 0.5, 0.5])

        def loss(theta):
            resid = self._objective_function(theta, probs_empirical)
            l = np.dot(resid, resid)
            return l

        result = minimize(loss, x0=initial_guess, method="L-BFGS-B")

        # if self.verbose:
        #     print(f"True probabilities: ", probs_true.round(3))
        #     print("Empirical probabilities: ", probs_empirical.round(3))

        #     est_probs = self._estimate_probs(result.x)
        #     print("Estimated probabilities: ", est_probs.round(3))

        self.A_hat, self.B_hat, self.C_hat = result.x

        if self.verbose:
            print(
                f"True parameters: {self.params['A']:.2f}, {self.params['B']:.2f}, {self.params['C']:.2f}"
            )
            print(
                f"Fitted parameters: {self.A_hat:.2f}, {self.B_hat:.2f}, {self.C_hat:.2f}"
            )
