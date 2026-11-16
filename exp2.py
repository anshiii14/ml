import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv("mushrooms.csv")

target = "class"

def entropy(data):
    counts = data[target].value_counts()
    total = len(data)
    value = 0

    for count in counts:
        p = count / total
        value = value - p * math.log2(p)

    return value

def information_gain(data, attribute):

    total_entropy = entropy(data)
    entropyaftersplit = 0

    for value in data[attribute].unique():

        subset = data[data[attribute] == value]
        weight = len(subset) / len(data)
        entropyaftersplit += weight * entropy(subset)

    return total_entropy - entropyaftersplit

def id3(data, attributes):

    classes = data[target]

    if len(classes.unique()) == 1:
        return classes.iloc[0]

    if len(attributes) == 0:
        return classes.mode()[0]

    print("\nCurrent entropy =", round(entropy(data), 5))

    gains = {}

    print("Gain:")

    for attribute in attributes:
        gains[attribute] = information_gain(data, attribute)
        print(attribute, "=", round(gains[attribute], 5))

    best_attribute = max(gains, key=gains.get)

    print("Selected attribute:", best_attribute)

    decisiontree = {best_attribute: {}}

    remaining_attributes = []

    for attribute in attributes:
        if attribute != best_attribute:
            remaining_attributes.append(attribute)

    for value in data[best_attribute].unique():

        subset = data[data[best_attribute] == value]

        if len(subset) == 0:
            decisiontree[best_attribute][value] = classes.mode()[0]

        else:
            decisiontree[best_attribute][value] = id3(
                subset.drop(columns=[best_attribute]),
                remaining_attributes
            )

    return decisiontree

def predict(decisiontree, row):

    if type(decisiontree) != dict:
        return decisiontree

    root = list(decisiontree.keys())[0]

    value = row[root]

    return predict(decisiontree[root][value], row)

def print_rules(decisiontree, rule=""):

    if type(decisiontree) != dict:
        print(rule + " THEN Class = " + decisiontree)
        return

    root = list(decisiontree.keys())[0]

    for value in decisiontree[root]:

        if rule == "":
            new_rule = "IF " + root + " = " + str(value)
        else:
            new_rule = rule + " AND " + root + " = " + str(value)

        print_rules(decisiontree[root][value], new_rule)

train_data, test_data = train_test_split(
    data,
    test_size=0.2,
    random_state=42,
    stratify=data[target]
)

attributes = list(train_data.columns)
attributes.remove(target)

decisiontree = id3(train_data, attributes)

print("\nGenerated decision tree")
print(decisiontree)

correct = 0

for _, row in test_data.iterrows():

    prediction = predict(decisiontree, row)

    if prediction == row[target]:
        correct += 1

manual_accuracy = correct / len(test_data)

print("\nManual ID3 accuracy =", round(manual_accuracy * 100, 2), "%")

print("\nDecision rules")

print_rules(decisiontree)

encoded = data.copy()

for column in encoded.columns:
    encoder = LabelEncoder()
    encoded[column] = encoder.fit_transform(encoded[column])

X = encoded.drop(target, axis=1)
Y = encoded[target]

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42,
    stratify=Y
)

model = DecisionTreeClassifier(criterion="entropy")

model.fit(X_train, Y_train)

prediction = model.predict(X_test)

library_accuracy = accuracy_score(Y_test, prediction)

print("\nScikit-learn accuracy =", round(library_accuracy * 100, 2), "%")

print("\nComparison between manual and library:")

if manual_accuracy > library_accuracy:
    print("The manually implemented ID3 gives better accuracy.")
elif manual_accuracy < library_accuracy:
    print("The Scikit-learn decision tree gives slightly better accuracy because it is optimized.")
else:
    print("Both methods give similar accuracy.")


import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

data = pd.read_csv("mushrooms.csv")

target = "class"

def entropy(data):
    counts = data[target].value_counts()
    total = len(data)
    value = 0

    for count in counts:
        p = count / total
        value = value - p * math.log2(p)

    return value

def information_gain(data, attribute):

    total_entropy = entropy(data)
    entropy_after_split = 0

    for value in data[attribute].unique():
        subset = data[data[attribute] == value]
        weight = len(subset) / len(data)
        entropy_after_split += weight * entropy(subset)

    return total_entropy - entropy_after_split

def split_information(data, attribute):

    value = 0

    for item in data[attribute].unique():
        subset = data[data[attribute] == item]
        weight = len(subset) / len(data)

        if weight > 0:
            value = value - weight * math.log2(weight)

    return value

def gain_ratio(data, attribute):

    gain = information_gain(data, attribute)
    split = split_information(data, attribute)

    if split == 0:
        return 0

    return gain / split

def id3(data, attributes):

    classes = data[target]

    if len(classes.unique()) == 1:
        return classes.iloc[0]

    if len(attributes) == 0:
        return classes.mode()[0]

    print("\nCurrent Entropy =", round(entropy(data), 5))

    ratios = {}

    print("Gain Ratio:")

    for attribute in attributes:
        ratios[attribute] = gain_ratio(data, attribute)
        print(attribute, "=", round(ratios[attribute], 5))

    best_attribute = max(ratios, key=ratios.get)

    print("Selected attribute:", best_attribute)

    decisiontree = {best_attribute: {}}

    remaining_attributes = []

    for attribute in attributes:
        if attribute != best_attribute:
            remaining_attributes.append(attribute)

    for value in data[best_attribute].unique():

        subset = data[data[best_attribute] == value]

        if len(subset) == 0:
            decisiontree[best_attribute][value] = classes.mode()[0]

        else:
            decisiontree[best_attribute][value] = id3(
                subset.drop(columns=[best_attribute]),
                remaining_attributes
            )

    return decisiontree

def predict(decisiontree, row):

    if type(decisiontree) != dict:
        return decisiontree

    root = list(decisiontree.keys())[0]

    value = row[root]

    if value not in decisiontree[root]:
        return "p"

    return predict(decisiontree[root][value], row)

def print_rules(decisiontree, rule=""):

    if type(decisiontree) != dict:
        print(rule + " THEN Class = " + decisiontree)
        return

    root = list(decisiontree.keys())[0]

    for value in decisiontree[root]:

        if rule == "":
            new_rule = "IF " + root + " = " + str(value)
        else:
            new_rule = rule + " AND " + root + " = " + str(value)

        print_rules(decisiontree[root][value], new_rule)


train_data, test_data = train_test_split(
data,
test_size=0.2,
random_state=42,
stratify=data[target]
)

attributes = list(train_data.columns)
attributes.remove(target)

decisiontree = id3(train_data, attributes)

print("\nGenerated Decision Tree")
print(decisiontree)

correct = 0
actual = []
predicted = []

for _, row in test_data.iterrows():

    prediction = predict(decisiontree, row)

    actual.append(row[target])
    predicted.append(prediction)

    if prediction == row[target]:
        correct += 1

manual_accuracy = correct / len(test_data)

print("\nManual Gain Ratio Accuracy =", round(manual_accuracy * 100, 2), "%")

print("\nConfusion Matrix")
cm = confusion_matrix(actual, predicted, labels=["e", "p"])
print(cm)

print("\nDecision Rules")
print_rules(decisiontree)

print("\nObservation")
print("Gain Ratio reduces the bias towards attributes having many distinct values.")
print("The generated tree is similar to the ID3 tree because the mushroom dataset has highly informative attributes.")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

data = pd.read_csv("mushrooms.csv")

target = "class"


def gini(data):

    counts = data[target].value_counts()

    total = len(data)

    value = 1

    for count in counts:

        p = count / total

        value = value - p * p

    return value


def gini_index(data, attribute):

    after_split = 0

    for value in data[attribute].unique():
        subset = data[data[attribute] == value]
        weight = len(subset) / len(data)
        after_split += weight * gini(subset)

    return after_split

def id3(data, attributes):

    classes = data[target]

    if len(classes.unique()) == 1:
        return classes.iloc[0]

    if len(attributes) == 0:
        return classes.mode()[0]

    print("\nCurrent Gini =", round(gini(data), 5))

    scores = {}

    print("Gini Index:")

    for attribute in attributes:

        scores[attribute] = gini_index(data, attribute)

        print(attribute, "=", round(scores[attribute], 5))

    best_attribute = min(scores, key=scores.get)

    print("Selected attribute:", best_attribute)

    decisiontree = {best_attribute: {}}

    remaining_attributes = []

    for attribute in attributes:

        if attribute != best_attribute:

            remaining_attributes.append(attribute)

    for value in data[best_attribute].unique():

        subset = data[data[best_attribute] == value]

        if len(subset) == 0:

            decisiontree[best_attribute][value] = classes.mode()[0]

        else:

            decisiontree[best_attribute][value] = id3(
                subset.drop(columns=[best_attribute]),
                remaining_attributes
            )

    return decisiontree


def predict(decisiontree, row):

    if type(decisiontree) != dict:
        return decisiontree

    root = list(decisiontree.keys())[0]

    value = row[root]

    if value not in decisiontree[root]:
        return classes.mode()[0] if 'classes' in globals() else "p"

    return predict(decisiontree[root][value], row)


def print_rules(decisiontree, rule=""):

    if type(decisiontree) != dict:
        print(rule + " THEN Class = " + decisiontree)
        return

    root = list(decisiontree.keys())[0]

    for value in decisiontree[root]:

        if rule == "":
            new_rule = "IF " + root + " = " + str(value)
        else:
            new_rule = rule + " AND " + root + " = " + str(value)

        print_rules(decisiontree[root][value], new_rule)


train_data, test_data = train_test_split(
    data,
    test_size=0.2,
    random_state=42,
    stratify=data[target]
)

attributes = list(train_data.columns)
attributes.remove(target)

decisiontree = id3(train_data, attributes)

print("\nGenerated Decision Tree")
print(decisiontree)

correct = 0
actual = []
predicted = []

for _, row in test_data.iterrows():

    prediction = predict(decisiontree, row)

    actual.append(row[target])
    predicted.append(prediction)

    if prediction == row[target]:
        correct += 1

manual_accuracy = correct / len(test_data)

print("\nManual Gini Accuracy =", round(manual_accuracy * 100, 2), "%")

print("\nConfusion Matrix")

cm = confusion_matrix(actual, predicted, labels=["e", "p"])

print(cm)

print("\nDecision Rules")

print_rules(decisiontree)

print("\nObservation")
print("Gini Index measures impurity without using logarithms.")
print("The attribute with the minimum weighted Gini is selected for splitting.")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

data = pd.read_csv("mushrooms.csv")

target = "class"

def misclassification(data):

    counts = data[target].value_counts()

    total = len(data)

    maximum = counts.max() / total

    return 1 - maximum


def misclassification_error(data, attribute):

    after_split = 0

    for value in data[attribute].unique():

        subset = data[data[attribute] == value]

        weight = len(subset) / len(data)

        after_split += weight * misclassification(subset)

    return after_split


def id3(data, attributes):

    classes = data[target]

    if len(classes.unique()) == 1:
        return classes.iloc[0]

    if len(attributes) == 0:
        return classes.mode()[0]

    print("\nCurrent Misclassification Error =", round(misclassification(data), 5))

    scores = {}

    print("Weighted Misclassification Error:")

    for attribute in attributes:

        scores[attribute] = misclassification_error(data, attribute)

        print(attribute, "=", round(scores[attribute], 5))

    best_attribute = min(scores, key=scores.get)

    print("Selected Attribute:", best_attribute)

    decisiontree = {best_attribute: {}}

    remaining_attributes = []

    for attribute in attributes:

        if attribute != best_attribute:

            remaining_attributes.append(attribute)

    for value in data[best_attribute].unique():

        subset = data[data[best_attribute] == value]

        if len(subset) == 0:

            decisiontree[best_attribute][value] = classes.mode()[0]

        else:

            decisiontree[best_attribute][value] = id3(
                subset.drop(columns=[best_attribute]),
                remaining_attributes
            )

    return decisiontree


def predict(decisiontree, row):

    if type(decisiontree) != dict:
        return decisiontree

    root = list(decisiontree.keys())[0]

    value = row[root]

    if value not in decisiontree[root]:
        return "p"

    return predict(decisiontree[root][value], row)


def print_rules(decisiontree, rule=""):

    if type(decisiontree) != dict:
        print(rule + " THEN Class = " + decisiontree)
        return

    root = list(decisiontree.keys())[0]

    for value in decisiontree[root]:

        if rule == "":
            new_rule = "IF " + root + " = " + str(value)
        else:
            new_rule = rule + " AND " + root + " = " + str(value)

        print_rules(decisiontree[root][value], new_rule)


decisiontree = id3(train_data, attributes)

print("\nGenerated Decision Tree")
print(decisiontree)


print("\nConfusion Matrix")

cm = confusion_matrix(actual, predicted, labels=["e", "p"])

print(cm)

print("\nDecision Rules")

print_rules(decisiontree)


print("\nObservation")
print("Misclassification error measures impurity using only the majority class.")
print("The attribute with the minimum weighted Misclassification error is selected.")
print("This implementation follows the standard impurity minimization approach.")

