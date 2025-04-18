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
