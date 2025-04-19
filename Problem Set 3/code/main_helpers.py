from typing import Dict, Literal, Optional, Tuple


import numpy as np
from scipy.optimize import minimize


Approach = Literal["Rust", "Forward Simulation", "Analytical"]


def _run_cvf_iter(V0, V1, states, theta, beta):
    # Parameters
    mu, R = theta
    gamma = 0.5772  # Euler's constant
    # continuation value (shared across V0, V1)
    v_cont = np.logaddexp(V0, V1) + gamma
    # action 0: do not replace machine
    next_state = np.minimum(states + 1, 5)
    V0_next = mu * states + beta * v_cont[next_state - 1]
    # action 1: replace machine
    V1_next_val = R + beta * v_cont[0]
    V1_next = np.full_like(V1, V1_next_val)
    return V0_next, V1_next


def _convergence_check(x, x_new, tol):
    """
    Check for convergence.
    """
    return np.max(np.abs(x_new - x)) < tol


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
        V0 = np.zeros(len(states), dtype=float)
        V1 = np.zeros(len(states), dtype=float)

        # begin value function iteration
        for it in range(max_iter):
            # for each state a, update value function
            V0_next, V1_next = _run_cvf_iter(
                V0=V0,
                V1=V1,
                states=states,
                theta=(self.params["mu"], self.params["R"]),
                beta=self.params["beta"],
            )
            # check for convergence
            if _convergence_check(V0, V0_next, tol) and _convergence_check(
                V1, V1_next, tol
            ):
                V0, V1 = V0_next.copy(), V1_next.copy()
                if self.verbose:
                    print(f"Convergence reached after {it+1} iterations.")
                break
            V0, V1 = V0_next, V1_next

        # store results
        self.choice_prob = np.exp(V1) / (np.exp(V0) + np.exp(V1))
        self.V0 = V0
        self.V1 = V1
        if self.verbose:
            print("Converged Value Function:")
            for a, v0, v1, p in zip(states, self.V0, self.V1, self.choice_prob):
                print(f"a={a}:  V0={v0:8.4f}  V1={v1:8.4f}  P(replace)={p:6.3f}")

    # simulate data
    def run_data_simulation(self):
        # Parameters
        T = self.params["T"]
        states_t = np.empty(T, dtype=int)
        choices_t = np.empty(T, dtype=int)
        rng = np.random.default_rng(self.seed)
        a = 1  # start with a new machine
        for t in range(T):
            states_t[t] = a
            # draw independent draws from T1EV
            eps0 = rng.gumbel(loc=0.0, scale=1.0)
            eps1 = rng.gumbel(loc=0.0, scale=1.0)
            # calculate utilities to determine choice (i = 1 or 0)
            u_keep = self.V0[a - 1] + eps0
            u_replace = self.V1[a - 1] + eps1
            if u_replace > u_keep:
                choices_t[t] = 1  # replace
                a = 1
            else:
                choices_t[t] = 0  # keep
                a = a + 1 if a < 5 else 5

        if self.verbose:
            print("First 10 simulated states:", states_t[:10])
            print("First 10 simulated choices:", choices_t[:10])

        self.data = {"states": states_t, "choices": choices_t}


class MachineReplacementEstimation:
    def __init__(
        self,
        data: Dict[str, np.array],
        params: Dict[str, float],
        seed: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        if {"states", "choices"} - data.keys():
            raise ValueError("Data must contain 'states' and 'choices' keys.")
        self.data = data
        self.params = params
        self.verbose = verbose
        self.seed = np.random.default_rng(seed) if seed else np.random.default_rng()
        self._results = None

    def is_estimated(self) -> bool:
        return self._results is not None

    def get_theta_hat(self) -> Tuple[float, float]:
        if self.is_estimated():
            return tuple(np.round(self._results.x, 5))
        raise ValueError("Theta has not been estimated yet.")

    def estimate_theta(
        self,
        approach: Approach,
        initial_guess: Optional[np.array] = None,
        # F0=None,
        # F1=None,
        N_sim: int = 5_000,
        # T=None,
    ):
        initial_guess = (
            np.asarray(initial_guess, float)
            if initial_guess is not None
            else np.array([-0.5, -0.5])
        )
        # for forward simulation, we need to precompute forward sims
        sim_cache = None
        if approach == "Forward Simulation":
            replacement_prob = self._get_empirical_replacement_prob()
            sim_cache = self._precompute_forward_simulation(replacement_prob, N_sim)
            print("Precomputed forward simulation values.")
        # simulate the draws outside the parameter estimation for forward simulation
        # if approach == "Forward Simulation":
        #     replacement_freq = self.get_replacement_prob()
        #     sim_val_1 = [
        #         self.forward_simulation_draws(
        #             F0, F1, replacement_freq, N_sim, 1, age, T
        #         )
        #         for age in range(1, 6)
        #     ]
        #     sim_val_0 = [
        #         self.forward_simulation_draws(
        #             F0, F1, replacement_freq, N_sim, 0, age, T
        #         )
        #         for age in range(1, 6)
        #     ]
        # else:
        #     sim_val_0 = None
        #     sim_val_1 = None

        res = minimize(
            fun=self._log_likelihood,
            x0=initial_guess,
            args=(approach, sim_cache),
            method="L-BFGS-B",
        )
        if not res.success:
            raise RuntimeError(
                f"Optimization failed: {res.message} (status code: {res.status})"
            )
        self._results = res

    def _log_likelihood(
        self,
        theta: np.array,
        approach: Approach,
        sim_cache: Optional[Tuple[np.array, np.array]] = None,
        # T=None
    ):
        match approach:
            case "Rust":
                V0, V1 = self._run_value_function_iteration(theta)
            case "Forward Simulation":
                if sim_cache is None:
                    raise ValueError(
                        "Simulation cache is required for forward simulation."
                    )
                V0, V1 = self._run_forward_sim(theta, *sim_cache)
            case "Analytical":
                V0, V1 = self._est_analytical_approx(theta)
            case _:  # equivalent of else
                raise ValueError(f"Unknown approach: {approach}.")

        # if approach == "Rust":
        #     V0, V1 = self._run_value_function_iteration(theta)
        # if approach == "Forward Simulation":
        #     V0, V1 = self._forward_sim(theta, sim_val_0, sim_val_1, T)
        # if approach == "Analytical":
        #     V1 = np.zeros(5)
        #     beta = self.params["beta"]
        #     replacement_freq = self.get_replacement_prob()

        #     def AM_formula(theta, age, replacement_freq):
        #         replacement_freq = np.array(
        #             [value for value in replacement_freq.values()]
        #         )
        #         next_age = np.minimum(age + 1, 5)
        #         mu, R = theta
        #         out = (
        #             mu * age
        #             - R
        #             - beta
        #             * (
        #                 np.log(replacement_freq[next_age - 1])
        #                 - np.log(replacement_freq[0])
        #             )
        #         )
        #         return out

        #     V0 = np.array(
        #         [AM_formula(theta, age, replacement_freq) for age in range(1, 6)]
        #     )
        # avoid numerical issues by using logaddexp
        p_replace = np.exp(V1 - np.logaddexp(V0, V1))
        p_obs = np.where(
            self.data["choices"] == 1,
            p_replace[self.data["states"] - 1],
            1.0 - p_replace[self.data["states"] - 1],
        )
        # clip to avoid log(0)
        p_obs = np.clip(p_obs, 1e-12, 1.0)
        ll = np.log(p_obs).sum()
        # return negative log-likelihood since we want to minimize
        return -ll

    # ------------------------------------------------------------------
    # Approach‑specific value functions
    # ------------------------------------------------------------------

    # 1. Nested fixed point (Rust)
    def _run_value_function_iteration(
        self, theta: np.array, tol: float = 1e-6, max_iter: int = 1_000
    ):
        # initialize value function vector
        states = np.array([1, 2, 3, 4, 5])
        V0 = np.zeros(len(states))
        V1 = np.zeros(len(states))
        for it in range(max_iter):
            # for each state a, update value function
            V0_next, V1_next = _run_cvf_iter(
                V0=V0,
                V1=V1,
                states=states,
                theta=theta,
                beta=self.params["beta"],
            )
            # check for convergence
            if _convergence_check(V0, V0_next, tol) and _convergence_check(
                V1, V1_next, tol
            ):
                V0, V1 = V0_next.copy(), V1_next.copy()
                if self.verbose:
                    print(f"Convergence reached after {it+1} iterations.")
                break
            V0, V1 = V0_next, V1_next
        return V0, V1

    # 2. Forward simulation
    def _get_empirical_replacement_prob(self) -> np.array:
        return np.array(
            [
                np.mean(self.data["choices"][self.data["states"] == s])
                for s in np.unique(self.data["states"])
            ],
            dtype=float,
        )

    def _precompute_forward_simulation(
        self, replacement_prob: np.array, N_sim: int
    ) -> Tuple[np.array, np.array]:
        # beta_pows = self.params["beta"] ** np.arange(self.params["T"])[:, None, None]
        states = np.unique(self.data["states"])
        V0_sims = np.zeros((len(states), self.params["T"], 3))
        V1_sims = np.zeros((len(states), self.params["T"], 3))
        for age0 in states:
            for init_rep in (0, 1):
                ages, replacements, epsilons = self._simulate_paths(
                    replacement_prob,
                    N=N_sim,
                    a0=age0,
                    r0=init_rep,
                )
                util_stream = self._calc_flow_utilities(
                    theta=None, ages=ages, repl=replacements, eps=epsilons
                )
                # util_discounted = util_stream * beta_pows
                # util_avg = util_discounted.sum(axis=0).mean(axis=0)
                target = V1_sims if init_rep else V0_sims
                target[age0 - 1] = util_stream.mean(axis=1)
        return V0_sims, V1_sims

    def _simulate_paths(
        self, replacement_prob: np.array, N: int, a0: int, r0: int
    ) -> Tuple[np.array, np.array, np.array]:

        ages = np.zeros((self.params["T"], N), dtype=int)
        repl = np.zeros((self.params["T"], N), dtype=int)
        ages[0, :] = a0
        repl[0, :] = r0
        for t in range(1, self.params["T"]):
            ages[t] = np.where(repl[t - 1] == 1, 1, np.minimum(ages[t - 1] + 1, 5))
            probs = replacement_prob[ages[t] - 1]
            repl[t] = self.seed.binomial(1, probs)

        # Gumbel draws
        lag_ages = ages[:-1]
        probs = replacement_prob[lag_ages - 1]
        eps = 0.5772 - np.log(probs * repl[1:] + (1 - repl[1:]) * (1 - probs))
        eps = np.vstack((np.zeros((1, N)), eps))
        return ages, repl, eps

    def _calc_flow_utilities(
        self,
        theta: Optional[np.array],
        ages: np.array,
        repl: np.array,
        eps: np.array,
    ) -> np.array:
        if theta is not None:
            mu, R = theta
            return R * repl + mu * ages * (1 - repl) + eps
        return np.stack((ages * (1 - repl), repl, eps), axis=-1)

    def _run_forward_sim(self, theta, V0_sims, V1_sims) -> Tuple[np.array, np.array]:
        mu, R = theta
        beta_pows = self.params["beta"] ** np.arange(self.params["T"])
        beta_pows = beta_pows[None, :, None]

        # Broadcast beta_t over the time axis and collapse with .sum(axis=1)
        def _discount(cache: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            # cache shape: (5, T, 3)
            disc_age = (cache * beta_pows).sum(axis=1)  # → (5, 3)
            return disc_age[:, 0], disc_age[:, 1], disc_age[:, 2]

        a0_V0, rep_V0, eps_V0 = _discount(V0_sims)
        a0_V1, rep_V1, eps_V1 = _discount(V1_sims)

        V0 = mu * a0_V0 + R * rep_V0 + eps_V0
        V1 = mu * a0_V1 + R * rep_V1 + eps_V1
        return V0, V1

    # forward simulation function
    def forward_simulation_draws(
        self, F0, F1, replacement_freq, N_sim, init_replace, init_age, T
    ):
        # initialize
        transition_matrices = [F0, F1]
        age_init = np.ones(N_sim, dtype=int) * int(init_age)
        replacement_init = [np.zeros(N_sim), np.ones(N_sim)]
        age_outcomes = np.array([1, 2, 3, 4, 5])
        # convert from dict to array for indexing
        replacement_freq = np.array([value for value in replacement_freq.values()])

        def draw_age(age_state, replacement_state):
            # form the CDF
            draw = replacement_state + (1 - replacement_state) * np.minimum(
                age_state + 1, 5
            )
            return draw.astype(int)

        def draw_replacement(age_state, replacement_freq):
            replacement_probs = replacement_freq[age_state - 1]
            draw = np.random.binomial(n=1, p=replacement_probs)
            return draw

        ages = np.zeros((T, N_sim))
        replacements = np.zeros((T, N_sim))
        ages[0, :] = age_init
        replacements[0, :] = replacement_init[init_replace]
        age_draw = draw_age(age_init, replacement_init[init_replace])
        ages[1, :] = age_draw
        replacement_draw = draw_replacement(age_draw, replacement_freq)
        replacements[1, :] = replacement_draw
        for i in range(2, T):
            age_draw = draw_age(age_draw, replacement_draw)
            ages[i, :] = age_draw
            replacement_draw = draw_replacement(age_draw, replacement_freq)
            replacements[i, :] = replacement_draw

        # calculate epsilon
        # get array of replacement frequencies
        lagged_ages = ages[:-1, :]
        epsilons = 0.5772 - np.log(
            replacement_freq[lagged_ages.astype(int) - 1] * replacements[1:, :]
            + (1 - replacements[1:, :])
            * (1 - replacement_freq[lagged_ages.astype(int) - 1])
        )
        epsilons = np.vstack((np.zeros((1, N_sim)), epsilons))
        return (
            np.mean(ages, axis=1),
            np.mean(replacements, axis=1),
            np.mean(epsilons, axis=1),
        )

    def _forward_sim(self, theta, sim_val_0, sim_val_1, T):
        def _age_specific(self, theta, sim_val_0, sim_val_1, T, age):
            ages_0 = sim_val_0[age - 1][0]
            replacements_0 = sim_val_0[age - 1][1]
            epsilons_0 = sim_val_0[age - 1][2]
            ages_1 = sim_val_1[age - 1][0]
            replacements_1 = sim_val_1[age - 1][1]
            epsilons_1 = sim_val_1[age - 1][2]
            mu, R = theta
            betas = np.array([self.params["beta"] ** t for t in range(T)])
            utilities_0 = betas * (
                R * replacements_0 + mu * ages_0 * (1 - replacements_0) + epsilons_0
            )
            utilities_1 = betas * (
                R * replacements_1 + mu * ages_1 * (1 - replacements_1) + epsilons_1
            )
            return np.sum(utilities_0), np.sum(utilities_1)

        out = np.array(
            [
                _age_specific(self, theta, sim_val_0, sim_val_1, T, age)
                for age in range(1, 6)
            ]
        )
        V0 = out[:, 0]
        V1 = out[:, 1]
        return V0, V1
