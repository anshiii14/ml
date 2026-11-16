import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression

df = pd.read_csv("electricity_cost_dataset.csv")

df = df[['site area', 'structure type', 'water consumption',
         'utilisation rate', 'resident count', 'electricity cost']]

columns = ['site area', 'water consumption',
           'utilisation rate', 'resident count']

scaler = StandardScaler()
df[columns] = scaler.fit_transform(df[columns])

df = pd.get_dummies(df, columns=['structure type'], dtype=int)

X = df.drop('electricity cost', axis=1)
y = df['electricity cost']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values

W = np.zeros(X_train.shape[1])
B = 0

learning_rate = 0.01
epochs = 2000

errors = []

for i in range(epochs):

    prediction = np.dot(X_train, W) + B

    error = prediction - y_train

    loss = np.mean(error ** 2) / 2
    errors.append(loss)

    dW = (1 / len(X_train)) * np.dot(X_train.T, error)
    dB = (1 / len(X_train)) * np.sum(error)

    W = W - learning_rate * dW
    B = B - learning_rate * dB

prediction = np.dot(X_test, W) + B

print("Manual Linear Regression")
print("MSE:", mean_squared_error(y_test, prediction))

model = LinearRegression()
model.fit(X_train, y_train)

library_prediction = model.predict(X_test)

print("\nLibrary Linear Regression")
print("MSE:", mean_squared_error(y_test, library_prediction))

plt.plot(errors)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Error")
plt.show()