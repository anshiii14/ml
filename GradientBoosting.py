
import numpy as np

# Input dataset
X = np.array([1, 2, 3, 4, 5, 6])

# Actual output
y = np.array([10, 20, 30, 40, 50, 60])

# Initial prediction
prediction = np.ones(len(y)) * np.mean(y)

# Learning rate
learning_rate = 0.1

# Number of iterations
n_estimators = 5

print("Initial Prediction:")
print(prediction)

# Train Gradient Boosting
for iteration in range(n_estimators):

    # Calculate residual errors
    residual = y - prediction

    # Find best threshold
    best_error = float("inf")
    best_threshold = 3
    best_left = 0
    best_right = 0

    for threshold in range(1, 6):

        left = residual[X <= threshold]
        right = residual[X > threshold]

        left_value = np.mean(left)
        right_value = np.mean(right)

        # Predict residuals
        pred = np.where(
            X <= threshold,
            left_value,
            right_value
        )

        error = np.sum((residual - pred) ** 2)

        if error < best_error:
            best_error = error
            best_threshold = threshold
            best_left = left_value
            best_right = right_value

    # Update predictions
    new_prediction = np.where(
        X <= best_threshold,
        best_left,
        best_right
    )

    prediction += learning_rate * new_prediction

    print("\nIteration:", iteration + 1)
    print("Residual:", residual)
    print("Prediction:", prediction)

# Final result
print("\nFinal Predictions:")
print(np.round(prediction, 2))

print("Gradient Boosting completed")
