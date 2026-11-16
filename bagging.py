
import numpy as np

# Dataset
X = np.array([1, 2, 3, 4, 5, 6])
y = np.array([0, 0, 0, 1, 1, 1])

# Number of models
n_models = 5

# Test data
test = np.array([2, 5])

# Store predictions
all_predictions = []

# Train multiple models
for model in range(n_models):

    # Bootstrap sampling
    indices = np.random.choice(
        len(X), len(X), replace=True
    )

    x_sample = X[indices]
    y_sample = y[indices]

    # Find the best threshold
    best_error = float("inf")
    best_threshold = 3

    for threshold in range(1, 6):

        pred = np.where(
            x_sample > threshold, 1, 0
        )

        error = np.sum(pred != y_sample)

        if error < best_error:
            best_error = error
            best_threshold = threshold

    # Predict test data
    prediction = np.where(
        test > best_threshold, 1, 0
    )

    all_predictions.append(prediction)

    print("Model:", model + 1)
    print("Threshold:", best_threshold)
    print("Prediction:", prediction)

# Combine predictions
all_predictions = np.array(all_predictions)

# Majority voting
final_prediction = []

for j in range(len(test)):

    votes = np.bincount(all_predictions[:, j])

    final_prediction.append(np.argmax(votes))

print("\nFinal Bagging Predictions:")
print(final_prediction)

print("Bagging completed")
