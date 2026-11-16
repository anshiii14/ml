import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("mushrooms.csv")

df.replace("?", np.nan, inplace=True)
df.fillna(df.mode().iloc[0], inplace=True)

df.drop_duplicates(inplace=True)

y = df["class"]
X = df.drop("class", axis=1)

y = y.map({"e": 0, "p": 1})

X = pd.get_dummies(X, drop_first=True, dtype=int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values

W = np.zeros(X_train.shape[1])
B = 0

learning_rate = 0.1
epochs = 5000

errors = []

for i in range(epochs):

    z = np.dot(X_train, W) + B

    prediction = 1 / (1 + np.exp(-z))

    prediction = np.clip(prediction, 1e-15, 1 - 1e-15)

    loss = -np.mean(
        y_train * np.log(prediction)
        + (1 - y_train) * np.log(1 - prediction)
    )

    errors.append(loss)

    error = prediction - y_train

    dW = np.dot(X_train.T, error) / len(X_train)
    dB = np.sum(error) / len(X_train)

    W = W - learning_rate * dW
    B = B - learning_rate * dB

z = np.dot(X_test, W) + B

probability = 1 / (1 + np.exp(-z))

y_pred = (probability >= 0.5).astype(int)

print("Manual Logistic Regression")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

library_pred = model.predict(X_test)

print("\nLibrary Logistic Regression")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, library_pred))

print("\nClassification Report:")
print(classification_report(y_test, library_pred))

plt.plot(errors)
plt.xlabel("Epoch")
plt.ylabel("Error")
plt.title("Logistic Regression Training Error")
plt.show()