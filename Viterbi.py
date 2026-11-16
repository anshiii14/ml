
import numpy as np

# Hidden states
states = ["Rainy", "Sunny"]

# Observations
observations = ["walk", "shop", "clean"]

# Initial probabilities
start_prob = np.array([0.6, 0.4])

# Transition probabilities
transition = np.array([
    [0.7, 0.3],
    [0.4, 0.6]
])

# Emission probabilities
emission = np.array([
    [0.1, 0.4, 0.5],
    [0.6, 0.3, 0.1]
])

# Given observation sequence
sequence = ["walk", "shop", "clean"]

# Convert observations into indexes
obs = [observations.index(x) for x in sequence]

# Viterbi algorithm
def viterbi(obs):
    n = len(states)
    T = len(obs)

    # Probability table
    dp = np.zeros((T, n))

    # Store previous states
    path = np.zeros((T, n), dtype=int)

    # Step 1: Initialization
    for i in range(n):
        dp[0, i] = start_prob[i] * emission[i, obs[0]]

    # Step 2: Recursion
    for t in range(1, T):
        for j in range(n):

            probabilities = (
                dp[t-1] * transition[:, j]
            )

            path[t, j] = np.argmax(probabilities)

            dp[t, j] = (
                np.max(probabilities) *
                emission[j, obs[t]]
            )

    # Step 3: Find final state
    last_state = np.argmax(dp[T-1])

    # Step 4: Backtracking
    best_path = [last_state]

    for t in range(T-1, 0, -1):
        last_state = path[t, last_state]
        best_path.append(last_state)

    # Reverse the path
    best_path.reverse()

    return [states[i] for i in best_path], np.max(dp[T-1])

# Execute algorithm
result, probability = viterbi(obs)

# Display result
print("Observation Sequence:", sequence)
print("Most Likely State Sequence:", result)
print("Probability:", probability)
