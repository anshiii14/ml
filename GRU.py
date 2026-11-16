
import numpy as np

# Input dataset
X = np.array([
    [[1], [2], [3]],
    [[2], [3], [4]],
    [[3], [4], [5]],
    [[4], [5], [6]]
], dtype=float)

y = np.array([[4], [5], [6], [7]], dtype=float)

# Normalize data
X = X / 7
y = y / 7

# Initialize weights
np.random.seed(42)

Wz = np.random.randn(1, 5) * 0.1
Wr = np.random.randn(1, 5) * 0.1
Wh = np.random.randn(1, 5) * 0.1

Uz = np.random.randn(5, 5) * 0.1
Ur = np.random.randn(5, 5) * 0.1
Uh = np.random.randn(5, 5) * 0.1

Wy = np.random.randn(5, 1) * 0.1

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -50, 50)))

# Training parameters
lr = 0.01
epochs = 500

# Training
for epoch in range(epochs):

    for sample in range(len(X)):

        h = np.zeros((1, 5))

        # Forward propagation
        for t in range(3):

            xt = X[sample, t].reshape(1, 1)

            # Update gate
            z = sigmoid(xt @ Wz + h @ Uz)

            # Reset gate
            r = sigmoid(xt @ Wr + h @ Ur)

            # Candidate state
            hn = np.tanh(xt @ Wh + (r*h) @ Uh)

            # New hidden state
            h = (1-z)*h + z*hn

        # Prediction
        pred = h @ Wy

        # Calculate error
        error = pred - y[sample].reshape(1, 1)

        # Update output weight
        Wy -= lr * (h.T @ error)

        # Update candidate weights (simplified)
        Wh -= lr * (xt.T @ (error @ Wy.T))
        Wz -= lr * (xt.T @ (error @ Wy.T))
        Wr -= lr * (xt.T @ (error @ Wy.T))

    if (epoch+1) % 100 == 0:
        print("Epoch:", epoch+1,
              "Error:", round(float(error[0, 0]**2), 5))

# Testing
test = [5, 6, 7]
h = np.zeros((1, 5))

for x in test:

    xt = np.array([[x/7]])

    z = sigmoid(xt @ Wz + h @ Uz)
    r = sigmoid(xt @ Wr + h @ Ur)

    hn = np.tanh(xt @ Wh + (r*h) @ Uh)

    h = (1-z)*h + z*hn

# Final prediction
prediction = (h @ Wy) * 7

print("\nActual Value: 8")
print("Predicted Value:", round(float(prediction[0, 0]), 2))
print("GRU Completed")
