import numpy as np
from scipy.optimize import minimize


def _run_cvf_iter(V0, V1, states, theta, beta):
    # Parameters
    mu, R = theta
    gamma = 0.5772  # Euler's constant
    # continuation value (shared across V0, V1)
    v_cont = np.log(np.exp(V0) + np.exp(V1)) + gamma
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
        V0, V1 = self._run_value_function_iteration(theta)
        p_replace = np.exp(V1) / (np.exp(V0) + np.exp(V1))
        p_obs = np.where(
            self.data["choices"] == 1,
            p_replace[self.data["states"] - 1],
            1.0 - p_replace[self.data["states"] - 1],
        )
        p_obs = np.clip(p_obs, 1e-12, 1.0)
        ll = np.log(p_obs).sum()
        # return negative log-likelihood since we want to minimize
        return -ll

    # value function iteration
    def _run_value_function_iteration(self, theta, tol=1e-6, max_iter=1000):
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
    
    def get_replacement_prob(self):
        if self.data is None:
            raise ValueError("Data has not been simulated yet.")
        # Count frequencies for states
        unique_states = np.unique(self.data["states"])
        replacement_freq = {
            state: np.mean(self.data["choices"][self.data["states"]==state]) for state in unique_states
        }
   
        return replacement_freq
    
    # forward simulation function
    N_sim = 1000 
    T = 20000
    F0  = np.array([[0, 1, 0, 0, 0], [0,0,1,0,0], [0,0,0,1,0], [0,0,0,0,1], [0,0,0,0,1]])
    F1 = np.array([[1,0,0,0,0],[1,0,0,0,0],[1,0,0,0,0],[1,0,0,0,0],[1,0,0,0,0]])
    replacement_init_0= np.zeros(N_sim)
    replacement_init_1= np.ones(N_sim)

    replacement_freq = get_replacement_prob(mod)
    def forward_simulation_draws(F0, F1, replacement_freq,N_sim, init_replace, T):
        #initialize
        transition_matrices = [F0, F1]
        age_init = np.ones(N_sim)
        replacement_init = [replacement_init_0, replacement_init_1]
        age_outcomes = np.array([1,2,3,4,5])
        
    
        def draw_age(x, age_state, replacement_state):
            draw = np.random.choice(a=age_outcomes,p= transition_matrices[int(replacement_state[x])][int(age_state[x])-1])
            return draw
        
        def draw_replacement(x, age_state, replacement_freq):
            draw = np.random.binomial(n=1, p = replacement_freq[int(age_state[x])])
            return draw 
        

        ages = np.zeros((T, N_sim))
        replacements = np.zeros((T, N_sim))
        ages[0,:]= age_init
        replacements[0,:] = replacement_init[init_replace]
        age_draw= np.array([draw_age(x, age_init, replacement_init[init_replace]) for x in range(N_sim)])
        ages[1,:]= age_draw
        replacement_draw = np.array([draw_replacement(x, age_draw, replacement_freq) for x in range(N_sim)])
        replacements[1,:] = replacement_draw
        for i in range(2,T):
            age_draw= np.array([draw_age(x, age_draw, replacement_draw) for x in range(N_sim)])
            ages[i,:]= age_draw
            replacement_draw = np.array([draw_replacement(x, age_draw, replacement_freq) for x in range(N_sim)])
            replacements[i,:] = replacement_draw
         
        # calculate epsilon
        # get array of replacement frequencies 
        replacement_array = []
        for a,b in replacement_freq.items(): 
            replacement_array.append(b)
        replacement_array = np.array(replacement_array)
     
        epsilons = .5772 - np.log(replacement_array[ages.astype(int)-1]*replacements +(1-replacements)*(1-replacement_array[ages.astype(int)-1]))

            
        return np.mean(ages, axis =1), np.mean(replacements, axis=1), np.mean(epsilons, axis =1)
    
    
ages_1, replacements_1, epsilons_1, = forward_simulation_draws(F0, F1, replacement_freq, N_sim, 0, T)
ages_0, replacements_0, epsilons_0 =  forward_simulation_draws(F0, F1, replacement_freq, N_sim, 1, T)

def forward_sim(theta, beta, ages_0, ages_1, replacements_0, replacements_1): 
    mu, R = theta 
    betas = np.array([beta**t for t in range(T)])
    utilities_0 = betas*(R*replacements_0 + mu*ages_0*(1-replacements_0) + epsilons_0)
    utilities_1 = betas*(R*replacements_1 + mu*ages_1*(1-replacements_1) + epsilons_1)
    utilities_0 = np.mean(axis=0, )
    # get replacement probabilities for each age 
