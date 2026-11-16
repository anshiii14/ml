import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("mushrooms.csv")

df.replace("?", np.nan, inplace=True)
df.fillna(df.mode().iloc[0], inplace=True)
df.drop_duplicates(inplace=True)

y = df["class"].map({"e": 0, "p": 1})
X = pd.get_dummies(df.drop("class", axis=1), drop_first=True, dtype=int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train = X_train.values.astype(float)
X_test = X_test.values.astype(float)
y_train = y_train.values
y_test = y_test.values

np.random.seed(42)

W1 = np.random.randn(X_train.shape[1], 16) * 0.01
W2 = np.random.randn(16, 8) * 0.01
W3 = np.random.randn(8, 1) * 0.01

b1 = np.zeros((1, 16))
b2 = np.zeros((1, 8))
b3 = np.zeros((1, 1))

learning_rate = 0.01
epochs = 3000

errors = []

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

for i in range(epochs):

    Z1 = np.dot(X_train, W1) + b1
    A1 = sigmoid(Z1)

    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)

    Z3 = np.dot(A2, W3) + b3
    A3 = sigmoid(Z3)

    A3_clip = np.clip(A3, 1e-15, 1 - 1e-15)

    loss = -np.mean(
        y_train.reshape(-1, 1) * np.log(A3_clip)
        + (1 - y_train.reshape(-1, 1)) * np.log(1 - A3_clip)
    )

    errors.append(loss)

    if i % 300 == 0:
        print("loss=", loss)

    d3 = A3 - y_train.reshape(-1, 1)

    dW3 = np.dot(A2.T, d3) / len(X_train)
    db3 = np.sum(d3, axis=0, keepdims=True) / len(X_train)

    d2 = np.dot(d3, W3.T) * A2 * (1 - A2)

    dW2 = np.dot(A1.T, d2) / len(X_train)
    db2 = np.sum(d2, axis=0, keepdims=True) / len(X_train)

    d1 = np.dot(d2, W2.T) * A1 * (1 - A1)

    dW1 = np.dot(X_train.T, d1) / len(X_train)
    db1 = np.sum(d1, axis=0, keepdims=True) / len(X_train)

    W3 -= learning_rate * dW3
    b3 -= learning_rate * db3

    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

Z1 = np.dot(X_test, W1) + b1
A1 = sigmoid(Z1)

Z2 = np.dot(A1, W2) + b2
A2 = sigmoid(Z2)

Z3 = np.dot(A2, W3) + b3
A3 = sigmoid(Z3)

y_pred = (A3 >= 0.5).astype(int).flatten()

print("Neural Network Accuracy:", accuracy_score(y_test, y_pred))
print("Neural Network Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

plt.plot(errors)
plt.xlabel("Epoch")
plt.ylabel("Error")
plt.title("Error vs Epoch")
plt.show()

tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)

tree_pred = tree.predict(X_test)

print("Decision Tree Accuracy:", accuracy_score(y_test, tree_pred))
print("Decision Tree Confusion Matrix:")
print(confusion_matrix(y_test, tree_pred))