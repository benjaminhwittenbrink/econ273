import numpy as np
from scipy.optimize import minimize


def _run_vfi_iter(V, states, theta, beta):
    # Parameters
    mu, R = theta
    gamma = 0.5772  # Euler's constant
    V_next = np.empty_like(V)
    for idx, a in enumerate(states):
        # replace machine (action 1), new state is 1
        v_replace = R + beta * V[0]
        # don't replace (action 0), new state is min{a + 1, 5}
        next_state = a + 1 if a < 5 else 5
        v_no_replace = mu * a + beta * V[next_state - 1]
        # logsum update (integrates over the T1EV shocks):
        V_next[idx] = np.log(np.exp(v_replace) + np.exp(v_no_replace)) + gamma
    return V_next


class MachineReplacementData:
    def __init__(self, params, seed, verbose=False):
        self.params = params
        self.seed = seed
        self.verbose = verbose

        np.random.seed(seed)
        self.V = None
        self.data = None

    def get_data(self):
        return self.data

    def get_data_frequencies(self):
        if self.data is None:
            raise ValueError("Data has not been simulated yet.")
        # Count frequencies for states
        unique_states = np.unique(self.data["states"])
        state_freq = {
            state: np.sum(self.data["states"] == state) for state in unique_states
        }
        # Count frequencies for choices
        unique_choices = np.unique(self.data["choices"])
        choice_freq = {
            choice: np.sum(self.data["choices"] == choice) for choice in unique_choices
        }
        return state_freq, choice_freq

    # value function iteration
    def run_value_function_iteration(self, tol=1e-6, max_iter=1000):
        # initialize value function vector
        states = np.array([1, 2, 3, 4, 5])
        V = np.zeros(len(states))
        for it in range(max_iter):
            # for each state a, update value function
            V_next = _run_vfi_iter(
                V=V,
                states=states,
                theta=(self.params["mu"], self.params["R"]),
                beta=self.params["beta"],
            )
            # check for convergence
            if np.max(np.abs(V_next - V)) < tol:
                V = V_next.copy()
                if self.verbose:
                    print(f"Convergence reached after {it+1} iterations.")
                break
            V = V_next.copy()
        if self.verbose:
            print("Converged Value Function:")
            for a, v in zip(states, V):
                print(f"State a = {a}: V({a}) = {v:.6f}")
        self.V = V

    # simulate data
    def run_data_simulation(self):
        # Parameters
        T = self.params["T"]
        states_t = np.empty(T, dtype=int)
        choices_t = np.empty(T, dtype=int)
        a = 1
        # loop over each period
        for t in range(T):
            states_t[t] = a
            # calculate utilities to determine choice (i = 1 or 0)
            # replace machine (action 1), new state is 1
            u_replace = self.params["R"] + self.params["beta"] * self.V[0]
            # don't replace (action 0), new state is min{a + 1, 5}
            next_a = a + 1 if a < 5 else 5
            u_no_replace = (
                self.params["mu"] * a + self.params["beta"] * self.V[next_a - 1]
            )
            # draw independent draws from T1EV
            eps_replace = np.random.gumbel(loc=0, scale=1)
            eps_no_replace = np.random.gumbel(loc=0, scale=1)
            # calculate utilties
            U1 = u_replace + eps_replace
            U0 = u_no_replace + eps_no_replace
            if U1 > U0:
                choices_t[t] = 1
                a = 1
            else:
                choices_t[t] = 0
                a = next_a
        if self.verbose:
            print("First 10 simulated states:", states_t[:10])
            print("First 10 simulated choices:", choices_t[:10])

        self.data = {"states": states_t, "choices": choices_t}


class MachineReplacementEstimation:
    def __init__(self, data, params, verbose=False):
        self.data = data
        self.params = params
        self.verbose = verbose
        self.results = None

    def get_theta(self):
        return np.round(self.results.x, 5)

    def estimate_theta(self, initial_guess=None):
        if self.data is None:
            raise ValueError("No data to estimate from.")
        # Parameters
        if initial_guess is None:
            initial_guess = np.array([-0.5, -0.5])

        res = minimize(
            fun=self._log_likelihood,
            x0=initial_guess,
            args=(self.data,),
            method="L-BFGS-B",
        )
        self.results = res

    def _log_likelihood(self, theta, tol=1e-6, max_iter=1000):
        # Parameters
        mu, R = theta
        beta = self.params["beta"]

        V = self._run_value_function_iteration(theta)

        ll = 0.0
        for a, choice in zip(self.data["states"], self.data["choices"]):
            # calculate utilties from each action
            u_replace = R + beta * V[0]
            next_a = a + 1 if a < 5 else 5
            u_no_replace = mu * a + beta * V[next_a - 1]
            # calculate probability of replacing
            exp_replace = np.exp(u_replace)
            exp_no_replace = np.exp(u_no_replace)
            p_replace = exp_replace / (exp_replace + exp_no_replace)
            # Determine likelihood contribution based on observed choice.
            if choice == 1:
                ll += np.log(p_replace)
            else:
                ll += np.log(1 - p_replace)
        return -ll

    # value function iteration
    def _run_value_function_iteration(self, theta, tol=1e-6, max_iter=1000):
        # initialize value function vector
        states = np.array([1, 2, 3, 4, 5])
        V = np.zeros(len(states))
        for it in range(max_iter):
            # for each state a, update value function
            V_next = _run_vfi_iter(
                V=V,
                states=states,
                theta=theta,
                beta=self.params["beta"],
            )
            # check for convergence
            if np.max(np.abs(V_next - V)) < tol:
                V = V_next.copy()
                if self.verbose:
                    print(f"Convergence reached after {it+1} iterations.")
                break
            V = V_next.copy()
        return V
