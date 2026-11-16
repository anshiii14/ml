
import numpy as np

# Input sequence
X = np.array([[1], [2], [3]])

# Initialize weights
np.random.seed(1)
Wx = np.random.randn(1, 2)
Wh = np.random.randn(2, 2)
Wy = np.random.randn(2, 1)

# Bias values
bh = np.zeros((1, 2))
by = np.zeros((1, 1))

# Activation function
def tanh(x):
    return np.tanh(x)

# RNN forward propagation
def rnn(sequence):
    h = np.zeros((1, 2))

    for t in range(len(sequence)):
        xt = sequence[t].reshape(1, 1)

        # Calculate hidden state
        h = tanh(xt @ Wx + h @ Wh + bh)

        print("Hidden state at step", t+1, ":", h)

    # Calculate output
    output = h @ Wy + by
    return output

# Run RNN
result = rnn(X)

print("\nFinal RNN Output:")
print(result)
