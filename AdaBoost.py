
import numpy as np

# Input dataset
X = np.array([1, 2, 3, 4, 5, 6])
y = np.array([-1, -1, -1, 1, 1, 1])

# Initialize equal weights
n = len(X)
weights = np.ones(n) / n

# Store weak learners
learners = []
alphas = []

# Train AdaBoost
for iteration in range(3):

    best_error = float("inf")
    best_pred = None
    best_threshold = None
    best_direction = None

    # Find the best decision stump
    for threshold in range(1, 6):

        for direction in [1, -1]:

            if direction == 1:
                pred = np.where(X > threshold, 1, -1)
            else:
                pred = np.where(X <= threshold, 1, -1)

            error = np.sum(weights * (pred != y))

            if error < best_error:
                best_error = error
                best_pred = pred
                best_threshold = threshold
                best_direction = direction

    # Calculate learner importance
    error = max(best_error, 1e-10)
    alpha = 0.5 * np.log((1-error) / error)

    # Update weights
    weights *= np.exp(-alpha * y * best_pred)
    weights /= np.sum(weights)

    learners.append((best_threshold, best_direction))
    alphas.append(alpha)

    print("Iteration:", iteration + 1)
    print("Error:", round(best_error, 4))
    print("Alpha:", round(alpha, 4))

# Make final prediction
test = np.array([2, 5])
final_score = np.zeros(len(test))

for (threshold, direction), alpha in zip(learners, alphas):

    if direction == 1:
        pred = np.where(test > threshold, 1, -1)
    else:
        pred = np.where(test <= threshold, 1, -1)

    final_score += alpha * pred

result = np.where(final_score >= 0, 1, -1)

print("\nFinal Predictions:", result)
print("AdaBoost completed")
