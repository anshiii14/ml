import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.naive_bayes import CategoricalNB

data = pd.read_csv("mushrooms.csv")

target = "class"

for column in data.columns:
    encoder = LabelEncoder()
    data[column] = encoder.fit_transform(data[column])

X = data.drop(target, axis=1)
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

classes = y_train.unique()

print("\nClasses:")
print(classes)

prior = {}

for c in classes:
    count = (y_train == c).sum()
    prior[c] = count / len(y_train)

print("\nPrior Probabilities:")

for c in classes:
    print("P(class =", c, ") =", prior[c])

likelihood = {}

for c in classes:
    likelihood[c] = {}
    class_data = X_train[y_train == c]

    for column in X_train.columns:
        likelihood[c][column] = {}
        values = X_train[column].unique()
        total = len(class_data)

        for value in values:
            count = (class_data[column] == value).sum()

            if count == 0:
                probability = (count + 1) / (total + len(values))
            else:
                probability = count / total

            likelihood[c][column][value] = probability

print("\nLikelihood calculations:")

feature = X_train.columns[0]

print("Feature:", feature)

for c in classes:
    print("\nClass:", c)

    for value in likelihood[c][feature]:
        probability = likelihood[c][feature][value]
        print("P(", feature, "=", value, "| class =", c, ") =", probability)

predictions = []

for index, row in X_test.iterrows():
    probabilities = {}

    for c in classes:
        probability = math.log(prior[c])

        for column in X_test.columns:
            value = row[column]
            probability += math.log(likelihood[c][column][value])

        probabilities[c] = probability

    prediction = max(probabilities, key=probabilities.get)
    predictions.append(prediction)

print("\nOur Naive Bayes Model")

cm = confusion_matrix(y_test, predictions)
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)

print("\nConfusion matrix:")
print(cm)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

library_model = CategoricalNB()
library_model.fit(X_train, y_train)

library_predictions = library_model.predict(X_test)

print("\nStandard library Naive Bayes")

cm_library = confusion_matrix(y_test, library_predictions)
library_accuracy = accuracy_score(y_test, library_predictions)
library_precision = precision_score(y_test, library_predictions)
library_recall = recall_score(y_test, library_predictions)

print("\nConfusion matrix:")
print(cm_library)
print("Accuracy:", library_accuracy)
print("Precision:", library_precision)
print("Recall:", library_recall)

print("\nComparison")

print("Model accuracy:", accuracy)
print("Library model accuracy:", library_accuracy)
print("Model precision:", precision)
print("Library model precision:", library_precision)
print("Model recall:", recall)
print("Library model recall:", library_recall)
