
import numpy as np

# Hidden states
states = ["Sunny", "Rainy"]

# Observations
observations = ["Walk", "Shop", "Clean"]

# Initial probabilities
start_prob = np.array([0.6, 0.4])

# Transition probabilities
trans_prob = np.array([
    [0.7, 0.3],
    [0.4, 0.6]
])

# Emission probabilities
emit_prob = np.array([
    [0.6, 0.3, 0.1],
    [0.1, 0.4, 0.5]
])

# Observation sequence
obs = [0, 1, 2]

# Initial probability
prob = start_prob * emit_prob[:, obs[0]]

print("Initial probabilities:")
print(prob)

# Calculate probability for each observation
for t in range(1, len(obs)):

    new_prob = np.zeros(2)

    for j in range(2):

        for i in range(2):

            new_prob[j] += (
                prob[i] *
                trans_prob[i][j]
            )

        new_prob[j] *= emit_prob[j][obs[t]]

    prob = new_prob

    print("Step", t+1, ":", prob)

# Final probability
total_prob = np.sum(prob)

print("\nFinal Probability:", total_prob)

print("HMM execution completed")
