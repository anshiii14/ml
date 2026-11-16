
import numpy as np

# Hidden states
states = ["Sunny", "Rainy"]

# Initial probabilities
start = np.array([0.6, 0.4])

# Transition probabilities
trans = np.array([
    [0.7, 0.3],
    [0.4, 0.6]
])

# Emission probabilities
emit = np.array([
    [0.8, 0.2],
    [0.3, 0.7]
])

# Observation sequence
obs = [0, 1, 0]

# Number of states and observations
N = len(states)
T = len(obs)

# Create probability table
dp = np.zeros((T, N))

# Create backtracking table
path = np.zeros((T, N), dtype=int)

# First observation
dp[0] = start * emit[:, obs[0]]

print("Initial Probability:")
print(dp[0])

# Calculate probabilities
for t in range(1, T):

    for j in range(N):

        # Calculate previous probabilities
        values = dp[t-1] * trans[:, j]

        # Find maximum probability
        best_previous = np.argmax(values)

        # Store best previous state
        path[t, j] = best_previous

        # Calculate current probability
        dp[t, j] = (
            values[best_previous] *
            emit[j, obs[t]]
        )

    print("\nStep:", t+1)
    print(dp[t])

# Find final state
last_state = np.argmax(dp[T-1])

# Backtracking
best_path = [last_state]

for t in range(T-1, 0, -1):

    previous = path[t, best_path[0]]

    best_path.insert(0, previous)

# Display result
print("\nMost Likely Hidden States:")

for state in best_path:
    print(states[state])

print("\nBest State Sequence:", best_path)

print("Viterbi completed")
