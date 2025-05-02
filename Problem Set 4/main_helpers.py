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
        self.ALL_STATES = [(0, 0), (0, 1), (1, 0), (1, 1)]

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
        """
        Choose the next state based on the current state and the value of entering or exiting.
        """
        psi = self.psi_draw()
        phi = self.phi_draw()

        val_enter, val_exit = self._val_enter_exit(psi, phi, my_state, other_state)
        next_state = int(np.argmax([val_exit, val_enter]))
        return next_state, psi, phi

    def simulate_data(self, num_periods=1000):
        """
        Simulate data for the model.
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

    def _empirical_probs_and_counts(self):
        """
        Calculate empirical probabilities and counts for each state based on simulated data.
        """
        counts = {s: [0, 0] for s in self.ALL_STATES}  # [den, num]

        for t in range(1, len(self.states)):
            prev = self.states[t - 1]  #  s_{t-1}
            curr = self.states[t]  #  s_t   (actions = next state)

            # “success” for firm 2 ⇔ curr second component == 1
            success2 = curr[1]
            # success for firm 1  ⇔ curr first component  == 1
            success1 = curr[0]

            if prev == (0, 0):
                counts[(0, 0)][0] += 2
                counts[(0, 0)][1] += success1 + success2
            elif prev == (0, 1):
                counts[(1, 0)][0] += 1
                counts[(1, 0)][1] += success1  # firm 1 moves
                counts[(0, 1)][0] += 1
                counts[(0, 1)][1] += success2  # firm 2 moves
            elif prev == (1, 0):
                counts[(0, 1)][0] += 1
                counts[(0, 1)][1] += success1
                counts[(1, 0)][0] += 1
                counts[(1, 0)][1] += success2
            elif prev == (1, 1):
                counts[(1, 1)][0] += 2
                counts[(1, 1)][1] += success1 + success2

        p_hat = {s: counts[s][1] / counts[s][0] for s in self.ALL_STATES}
        return p_hat, counts

    def _solve_equilibrium(self, theta):
        """
        Solve the equilibrium for the given parameters.
        """
        A, B, C = theta
        sol = opt.root(
            self._optimize_system_func,
            np.ones(8),
            method="hybr",
            options={"xtol": 1e-8},
            args=(A, B, C),
        )
        if not sol.success:
            raise RuntimeError(sol.message)
        V00, V01, V10, V11, p00, p01, p10, p11 = sol.x
        p = {(0, 0): p00, (0, 1): p01, (1, 0): p10, (1, 1): p11}
        V = {(0, 0): V00, (0, 1): V01, (1, 0): V10, (1, 1): V11}
        return p, V

    def _moment_vector(self, theta, p_hat):
        """
        Calculate the moment vector for the GMM estimation.
        """
        p_sim, _ = self._solve_equilibrium(theta)
        return np.array([p_hat[s] - p_sim[s] for s in self.ALL_STATES])

    def _numerical_jacobian(self, theta, p_hat, eps=1e-6):
        """
        Calculate the numerical Jacobian of the moment vector.
        """
        g0 = self._moment_vector(theta, p_hat)
        jac = np.zeros((len(g0), len(theta)))
        for j in range(len(theta)):
            step = np.zeros_like(theta, float)
            step[j] = eps
            jac[:, j] = (
                self._moment_vector(theta + step, p_hat)
                - self._moment_vector(theta - step, p_hat)
            ) / (2.0 * eps)
        return jac

    def _build_sigma(self, counts, p_hat):
        """
        Build the variance-covariance matrix for the GMM estimation.
        """
        Sigma = np.zeros((4, 4))
        for i, s in enumerate(self.ALL_STATES):
            n = counts[s][0]
            Sigma[i, i] = p_hat[s] * (1.0 - p_hat[s]) / n if n > 0 else 1e-8
        return Sigma

    def _efficient_weight_matrix(self, counts, p_hat):
        """
        Build the efficient weight matrix for the GMM estimation.
        """
        Sigma = self._build_sigma(counts, p_hat)
        # Since there are no g parameters, weighting matrix simplifies to:
        return 0.25 * np.linalg.inv(Sigma)

    def _gmm_variance_efficient(self, theta_hat, p_hat, counts):
        """
        Calculate the GMM variance-covariance matrix and standard errors.
        """
        D_theta = self._numerical_jacobian(theta_hat, p_hat)
        W_star = self._efficient_weight_matrix(counts, p_hat)
        vcov = np.linalg.inv(D_theta.T @ W_star @ D_theta)
        se = np.sqrt(np.diag(vcov))
        return vcov, se

    def _loss_function(self, theta, p_hat, W=np.eye(4)):
        """
        Calculate the loss function for the GMM estimation.
        """
        resid = self._moment_vector(theta, p_hat)
        return resid.T @ W @ resid

    def _asymptotic_least_squares_estimator(self):
        """
        Asymptotic least squares estimator for the model based on Pesendorder Schmidt-Dengler (2008).
        """
        p_hat, counts = self._empirical_probs_and_counts()

        # step‑1 Pesendorfer Schmidt‑Dengler least squares estimator
        res_step1 = minimize(
            lambda th: self._loss_function(th, p_hat),
            x0=np.array([0.6, 0.5, 0.2]),
            method="L-BFGS-B",
            bounds=((1e-4, 0.999),) * 3,
        )
        if not res_step1.success:
            raise RuntimeError(res_step1.message)

        # step-2 Pesendorfer Schmidt‑Dengler least squares estimator (with optimal weight matrix)
        weight_matrix = self._efficient_weight_matrix(counts, p_hat)
        res_step2 = minimize(
            lambda th: self._loss_function(th, p_hat, W=weight_matrix),
            x0=res_step1.x,
            method="L-BFGS-B",
            bounds=((1e-4, 0.999),) * 3,
        )
        if not res_step2.success:
            raise RuntimeError(res_step2.message)

        self.A_hat, self.B_hat, self.C_hat = theta_hat = res_step2.x

        # step‑3 PSD‑GMM standard errors
        self.vcov, self.se = self._gmm_variance_efficient(theta_hat, p_hat, counts)

    def estimate_model(self):
        """
        Estimate the model parameters using GMM.
        """
        if self.states is None:
            self.simulate_data()

        self._asymptotic_least_squares_estimator()

        if self.verbose:
            print(
                f"\nTrue (A,B,C) : {self.params['A']:.3f}, "
                f"{self.params['B']:.3f}, {self.params['C']:.3f}"
            )
            print(
                f" Est. (A,B,C): {self.A_hat:.3f}, "
                f"{self.B_hat:.3f}, {self.C_hat:.3f}"
            )
            print(" s.e.        :", "  ".join(f"{x:.3f}" for x in self.se))
