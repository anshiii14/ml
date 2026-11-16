
import numpy as np

# Step 1: Training dataset
X = np.array([
    [[1], [2], [3]],
    [[2], [3], [4]],
    [[3], [4], [5]],
    [[4], [5], [6]]
], dtype=float)

y = np.array([[4], [5], [6], [7]], dtype=float)

# Step 2: Normalize data
X = X / 7
y = y / 7

# Step 3: Initialize weights
np.random.seed(42)
Wx = np.random.randn(1, 5) * 0.1
Wh = np.random.randn(5, 5) * 0.1
Wy = np.random.randn(5, 1) * 0.1

# Step 4: Initialize bias
bh = np.zeros((1, 5))
by = np.zeros((1, 1))

# Step 5: Learning parameters
lr = 0.01
epochs = 500

# Step 6: Training process
for epoch in range(epochs):

    total_loss = 0

    for sample in range(len(X)):

        h = np.zeros((1, 5))
        hidden = []

        # Forward propagation
        for t in range(3):

            xt = X[sample, t].reshape(1, 1)

            h = np.tanh(xt @ Wx + h @ Wh + bh)

            hidden.append(h)

        # Output prediction
        pred = h @ Wy + by

        # Calculate error
        error = pred - y[sample].reshape(1, 1)

        total_loss += np.mean(error ** 2)

        # Backpropagation
        dWy = h.T @ error
        dby = error.copy()

        dh = error @ Wy.T

        dWx = np.zeros_like(Wx)
        dWh = np.zeros_like(Wh)
        dbh = np.zeros_like(bh)

        # Backpropagation through time
        for t in range(2, -1, -1):

            dh_raw = dh * (1 - hidden[t] ** 2)

            xt = X[sample, t].reshape(1, 1)

            hp = np.zeros((1, 5)) if t == 0 else hidden[t-1]

            dWx += xt.T @ dh_raw
            dWh += hp.T @ dh_raw
            dbh += dh_raw

            dh = dh_raw @ Wh.T

        # Update weights
        Wx -= lr * dWx
        Wh -= lr * dWh
        Wy -= lr * dWy

        # Update bias
        bh -= lr * dbh
        by -= lr * dby

    # Display loss
    if (epoch + 1) % 100 == 0:
        print("Epoch:", epoch+1,
              "Loss:", round(total_loss/len(X), 5))

# Step 7: Testing
test = np.array([[[5], [6], [7]]]) / 7

h = np.zeros((1, 5))

for t in range(3):
    xt = test[0, t].reshape(1, 1)
    h = np.tanh(xt @ Wx + h @ Wh + bh)

# Step 8: Final prediction
prediction = (h @ Wy + by) * 7

print("\nActual Value: 8")
print("Predicted Value:", round(prediction[0, 0], 2))
print("RNN Training Completed")
