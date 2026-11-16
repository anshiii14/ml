
import numpy as np

# Input sequence
X = [1, 2, 3, 4]

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Initial cell state
c = 0.0

# Initial hidden state
h = 0.0

# Process each input
for x in X:

    # Forget gate
    f = sigmoid(x * 0.2)

    # Input gate
    i = sigmoid(x * 0.3)

    # Candidate memory
    g = np.tanh(x * 0.5)

    # Update cell state
    c = f*c + i*g

    # Output gate
    o = sigmoid(x * 0.4)

    # Update hidden state
    h = o * np.tanh(c)

    print("\nInput:", x)

    print("Forget Gate:", round(f, 4))

    print("Input Gate:", round(i, 4))

    print("Output Gate:", round(o, 4))

    print("Cell State:", round(c, 4))

    print("Hidden State:", round(h, 4))

# Final output
print("\nFinal LSTM Output:")
print(round(h, 4))

# Test sequence
test = [2, 3, 4]

c = 0.0
h = 0.0

for x in test:

    f = sigmoid(x * 0.2)
    i = sigmoid(x * 0.3)
    g = np.tanh(x * 0.5)

    c = f*c + i*g

    o = sigmoid(x * 0.4)

    h = o * np.tanh(c)

print("\nTest Prediction:")
print(round(h, 4))

print("LSTM execution completed")
