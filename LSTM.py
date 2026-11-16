
import numpy as np

# Input sequence
X = np.array([[1], [2], [3], [4]], dtype=float)

# Initialize weights
np.random.seed(1)

Wx = np.random.randn(1, 8) * 0.1
Wh = np.random.randn(2, 8) * 0.1
b = np.zeros((1, 8))

Wy = np.random.randn(2, 1) * 0.1
by = np.zeros((1, 1))

# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

# LSTM function
def lstm(sequence):
    h = np.zeros((1, 2))
    c = np.zeros((1, 2))

    for t in range(len(sequence)):

        # Current input
        xt = sequence[t].reshape(1, 1)

        # Calculate gates
        gates = xt @ Wx + h @ Wh + b

        # Forget gate
        f = sigmoid(gates[:, 0:2])

        # Input gate
        i = sigmoid(gates[:, 2:4])

        # Candidate cell state
        g = tanh(gates[:, 4:6])

        # Output gate
        o = sigmoid(gates[:, 6:8])

        # Update cell state
        c = f * c + i * g

        # Update hidden state
        h = o * tanh(c)

        print("Step", t+1)
        print("Forget Gate:", f)
        print("Input Gate:", i)
        print("Cell State:", c)
        print("Output Gate:", o)
        print("Hidden State:", h)
        print()

    # Final prediction
    output = h @ Wy + by
    return output

# Execute LSTM
result = lstm(X)

print("Final LSTM Output:")
print(result)
