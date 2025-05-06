import numpy as np

# Parameters
mu = -1
R = -3
beta = 0.9
gamma = 0.5775  # Euler's constant
states = np.array([1, 2, 3, 4, 5])

# initialize value function vector
V = np.zeros(len(states))

# Set tolerance level and maximum iterations for convergence.
tol = 1e-6
max_iter = 1000

# VFI
for it in range(max_iter):
    V_next = np.empty_like(V)
    # for each state a, update VFI
    for idx, a in enumerate(states):
        # replace machine (action 1), new state is 1
        v_replace = R + beta * V[0]

        # don't replace (action 0), new state is min{a + 1, 5}
        next_state = a + 1 if a < 5 else 5
        v_no_replace = mu * a + beta * V[next_state - 1]

        # Logsum update (integrates over the T1EV shocks):
        V_next[idx] = np.log(np.exp(v_replace) + np.exp(v_no_replace)) + gamma

    # Check for convergence
    if np.max(np.abs(V_next - V)) < tol:
        V = V_next.copy()
        print(f"Convergence reached after {it+1} iterations.")
        break

    V = V_next.copy()

print("Converged Value Function:")
for a, v in zip(states, V):
    print(f"State a = {a}: V({a}) = {v:.6f}")
