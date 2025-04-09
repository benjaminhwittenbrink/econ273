import numpy as np


class RustSimulation:
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
            V_next = self._run_vfi_iter(V, states)
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

    def _run_vfi_iter(self, V, states):
        # Parameters
        mu = self.params["mu"]
        R = self.params["R"]
        beta = self.params["beta"]
        gamma = self.params["gamma"]
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
