
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier

np.random.seed(42)

df = pd.read_csv("heart_cleveland_upload.csv")
df.replace("?", np.nan, inplace=True)
df = df.apply(pd.to_numeric, errors="coerce")
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

X = df.drop(columns=["condition"]).values.astype(float)
y = df["condition"].values.astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

Y = np.where(y_train == 0, -1, 1)
n, m = X_train.shape

W = np.zeros(m)
B = 0.0
alpha = 0.01
C = 5

for epoch in range(1000):
    for i in range(n):
        value = np.dot(X_train[i], W) + B
        loss = max(0, 1 - Y[i] * value)

        if loss==0:
            W = W-alpha*W
        else:
            W = W - alpha*(W-C*Y[i]*X_train[i])
            B = B + alpha*C*Y[i]

prediction = np.where(np.dot(X_test, W) + B > 0, 1, 0)

print("Linear SVM")
print(classification_report(y_test, prediction, zero_division=0))

scores = np.dot(X_train, W) + B
support_vectors = X_train[Y * scores <= 1]

print("Linear support vectors:")
print(support_vectors)
print("Number:", len(support_vectors))


gamma = 0.1
C = 5
tolerance = 0.001

def kernel(A, B):
    d = np.sum(A**2, axis=1).reshape(-1, 1)
    e = np.sum(B**2, axis=1)
    return np.exp(-gamma * (d + e - 2 * np.dot(A, B.T)))

K = kernel(X_train, X_train)
a = np.zeros(n)
b = 0.0
passes = 0
count = 0

while passes < 20 and count < 500:
    changed = 0

    for i in range(n):
        f1 = np.sum(a * Y * K[:, i]) + b
        E1 = f1 - Y[i]

        if (Y[i] * E1 < -tolerance and a[i] < C) or (
            Y[i] * E1 > tolerance and a[i] > 0
        ):
            j = np.random.randint(n)
            while j == i:
                j = np.random.randint(n)

            f2 = np.sum(a * Y * K[:, j]) + b
            E2 = f2 - Y[j]

            old_i = a[i]
            old_j = a[j]

            if Y[i] != Y[j]:
                L = max(0, a[j] - a[i])
                H = min(C, C + a[j] - a[i])
            else:
                L = max(0, a[i] + a[j] - C)
                H = min(C, a[i] + a[j])

            eta = 2 * K[i, j] - K[i, i] - K[j, j]

            if L == H or eta >= 0:
                continue

            a[j] -= Y[j] * (E1 - E2) / eta
            a[j] = np.clip(a[j], L, H)

            if abs(a[j] - old_j) < 0.00001:
                a[j] = old_j
                continue

            a[i] += Y[i] * Y[j] * (old_j - a[j])

            b1 = b - E1 - Y[i] * (a[i] - old_i) * K[i, i] - Y[j] * (a[j] - old_j) * K[i, j]
            b2 = b - E2 - Y[i] * (a[i] - old_i) * K[i, j] - Y[j] * (a[j] - old_j) * K[j, j]

            if 0 < a[i] < C:
                b = b1
            elif 0 < a[j] < C:
                b = b2
            else:
                b = (b1 + b2) / 2

            changed += 1

    if changed == 0:
        passes += 1
    else:
        passes = 0

    count += 1

mask = a > 0.00001
sv2 = X_train[mask]
sv_y = Y[mask]
sv_a = a[mask]

pred3 = np.where(np.dot(kernel(X_test, sv2), sv_a * sv_y) + b > 0, 1, 0)

print("RBF KERNEL SVM")
print(classification_report(y_test, pred3, zero_division=0))
print("RBF support vectors:")
print(sv2)
print("Number:", len(sv2))