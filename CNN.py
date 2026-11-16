
import numpy as np

# Input image (5x5)
image = np.array([
    [1, 0, 1, 0, 1],
    [0, 1, 0, 1, 0],
    [1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1],
    [1, 0, 0, 1, 0]
], dtype=float)

# 3x3 filter (kernel)
kernel = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
], dtype=float)

# Convolution operation
def convolution(img, filt):
    h, w = img.shape
    kh, kw = filt.shape
    output = np.zeros((h-kh+1, w-kw+1))

    for i in range(h-kh+1):
        for j in range(w-kw+1):
            region = img[i:i+kh, j:j+kw]
            output[i, j] = np.sum(region * filt)

    return output

# ReLU activation
def relu(x):
    return np.maximum(0, x)

# Max pooling (2x2)
def max_pooling(x):
    h, w = x.shape
    result = np.zeros((h//2, w//2))

    for i in range(0, h-1, 2):
        for j in range(0, w-1, 2):
            result[i//2, j//2] = np.max(x[i:i+2, j:j+2])

    return result

# Flatten the feature map
def flatten(x):
    return x.flatten()

# Fully connected output
def predict(features):
    weights = np.ones(len(features))
    score = np.dot(features, weights)
    return "Class 1" if score > 0 else "Class 0"

# CNN execution
conv = convolution(image, kernel)
activated = relu(conv)
pooled = max_pooling(activated)
features = flatten(pooled)
result = predict(features)

print("Input Image:")
print(image)
print("\nAfter Convolution:")
print(conv)
print("\nAfter ReLU:")
print(activated)
print("\nAfter Max Pooling:")
print(pooled)
print("\nFlattened Features:")
print(features)
print("\nPredicted Output:", result)
