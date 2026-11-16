
import numpy as np

# Input image
image = np.array([
    [1, 0, 1, 0, 1],
    [0, 1, 0, 1, 0],
    [1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1],
    [1, 0, 0, 1, 0]
])

# 3x3 filter
kernel = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
])

# Convolution
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

# Max pooling
def max_pooling(x):
    h, w = x.shape
    result = np.zeros((h//2, w//2))

    for i in range(0, h-1, 2):
        for j in range(0, w-1, 2):
            result[i//2, j//2] = np.max(x[i:i+2, j:j+2])

    return result

# Execute CNN operations
conv = convolution(image, kernel)
activated = relu(conv)
pooled = max_pooling(activated)
features = pooled.flatten()

print("Convolution Output:")
print(conv)

print("\nAfter ReLU:")
print(activated)

print("\nAfter Max Pooling:")
print(pooled)

print("\nFlattened Features:")
print(features)
