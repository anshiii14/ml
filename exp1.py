import numpy as np

a = np.array([10, 20, 30, 40])
print("np.array()")
print(a)

b = np.zeros((3, 4))
print("\nnp.zeros()")
print(b)

c = np.ones((4, 3))
print("\nnp.ones()")
print(c)

d = np.empty((3, 3))
print("\nnp.empty()")
print(d)

e = np.eye(3)
print("\nnp.eye()")
print(e)

f = np.identity(3)
print("\nnp.identity()")
print(f)

g = np.arange(1, 15, 2)
print("\nnp.arange()")
print(g)

h = np.linspace(0, 30, 5)
print("\nnp.linspace()")
print(h)

i = np.logspace(0, 4, 4)
print("\nnp.logspace()")
print(i)

import numpy as np

a = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]])

print("Array")
print(a)

print("\nShape")
print(a.shape)

print("\nSize")
print(a.size)

print("\nNumber of Dimensions")
print(a.ndim)

print("\nData Type")
print(a.dtype)

print("\nItem Size")
print(a.itemsize)

import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])

print("Original Array")
print(a)

b = a.reshape(3, 2)
print("\nreshape()")
print(b)

c = a.flatten()
print("\nflatten()")
print(c)

d = a.ravel()
print("\nravel()")
print(d)

e = np.transpose(a)
print("\ntranspose()")
print(e)

print("\nT")
print(a.T)

f = np.array([[1, 2], [3, 4]])
f.resize((4, 1))
print("\nresize()")
print(f)

g = np.array([[[1, 2, 3], [4, 5, 6]],
              [[7, 8, 9], [10, 11, 12]]])

print("\nOriginal 3D Array")
print(g)

h = np.swapaxes(g, 0, 1)
print("\nswapaxes()")
print(h)

import numpy as np

a = np.array([16, 36, 4, 9, 25])

print("Array")
print(a)

print("\nsum()")
print(np.sum(a))

print("\nmean()")
print(np.mean(a))

print("\nmedian()")
print(np.median(a))

print("\nstd()")
print(np.std(a))

print("\nvar()")
print(np.var(a))

print("\nmin()")
print(np.min(a))

print("\nmax()")
print(np.max(a))

print("\nargmin()")
print(np.argmin(a))

print("\nargmax()")
print(np.argmax(a))

print("\nprod()")
print(np.prod(a))

print("\nsqrt()")
print(np.sqrt(a))

print("\nsquare()")
print(np.square(a))

print("\npower()")
print(np.power(a, 2))

b = np.array([1, 2, 3])

print("\nexp()")
print(np.exp(b))

print("\nlog()")
print(np.log(b))

print("\nlog10()")
print(np.log10(b))

c = np.array([-10, -5, 0, 5, 10])

print("\nabs()")
print(np.abs(c))

d = np.array([2.34, 5.67, 8.91])

print("\nround()")
print(np.round(d))

e = np.array([5, 15, 25, 35, 45])

print("\nclip()")
print(np.clip(e, 10, 30))

import numpy as np

a = np.array([10, 20, 30, 40, 50])

print("Array")
print(a)

print("\npercentile()")
print(np.percentile(a, 50))

print("\nquantile()")
print(np.quantile(a, 0.5))

b = np.array([1, 2, 3, 4, 5])
c = np.array([2, 4, 6, 8, 10])

print("\ncorrcoef()")
print(np.corrcoef(b, c))

print("\ncov()")
print(np.cov(b, c))

print("\nhistogram()")
hist, bins = np.histogram(a, bins=5)
print("Frequency:", hist)
print("Bins:", bins)

import numpy as np

print("random.seed()")
np.random.seed(10)
print("Seed set to 10")

print("\nrandom.rand()")
print(np.random.rand(2, 3))

print("\nrandom.randn()")
print(np.random.randn(2, 3))

print("\nrandom.randint()")
print(np.random.randint(1, 20, 5))

print("\nrandom.choice()")
print(np.random.choice([10, 20, 30, 40, 50], 3))

a = np.array([1, 2, 3, 4, 5])

np.random.shuffle(a)
print("\nshuffle()")
print(a)

print("\npermutation()")
print(np.random.permutation([1, 2, 3, 4, 5]))

import numpy as np

a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

print("Matrix A")
print(a)

print("\nMatrix B")
print(b)

print("\ndot()")
print(np.dot(a, b))

print("\nmatmul()")
print(np.matmul(a, b))

x = np.array([1, 2, 3])
y = np.array([4, 5, 6])

print("\ninner()")
print(np.inner(x, y))

print("\nouter()")
print(np.outer(x, y))

print("\ncross()")
print(np.cross(x, y))

print("\nlinalg.inv()")
print(np.linalg.inv(a))

print("\nlinalg.det()")
print(np.linalg.det(a))

print("\nlinalg.eig()")
values, vectors = np.linalg.eig(a)
print("Eigenvalues:")
print(values)
print("Eigenvectors:")
print(vectors)

print("\nlinalg.norm()")
print(np.linalg.norm(a))

c = np.array([[2, 1], [1, 3]])
d = np.array([8, 13])

print("\nlinalg.solve()")
print(np.linalg.solve(c, d))

import numpy as np

a = np.array([40, 10, 30, 20, 10, 50])

print("Array")
print(a)

print("\nsort()")
print(np.sort(a))

print("\nargsort()")
print(np.argsort(a))

print("\nunique()")
print(np.unique(a))

print("\nwhere()")
print(np.where(a > 25))

import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print("Array A")
print(a)

print("\nArray B")
print(b)

print("\nconcatenate()")
print(np.concatenate((a, b)))

print("\nstack()")
print(np.stack((a, b)))

print("\nvstack()")
print(np.vstack((a, b)))

print("\nhstack()")
print(np.hstack((a, b)))

c = np.array([10, 20, 30, 40, 50, 60])

print("\nsplit()")
print(np.split(c, 3))

import pandas as pd

data = {
    "ID": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112],
    "Name": ["Anshi", "Khushee", "Manya", "Krisha", "Karan", "Shreya", "Rohan", "Pooja", "Akshat", "Trupti", "Dev", "Anu"],
    "Age": [25, 24, 23, 25, 21, None, 22, 24, 28, 23, 27, None],
    "Department": ["IT", "ML", "IT", "ML", "SE", "PC", "IT", "PC", "ML", "Finance", "ML", "IT"],
    "Salary": [50000, 45000, 52000, 48000, None, 60000, 55000, 47000, 53000, 62000, 49000, 51000],
    "City": ["Mumbai", "Pune", "Delhi", "Mumbai", "Pune", "Chennai", "Bangalore", "Hyderabad", "Delhi", "Mumbai", "Pune", "Bangalore"],
    "Experience": [4, 2, 1, 3, 1, 4, 5, 2, 3, 2, 4, 3],
    "Gender": ["Male", "Female", "Male", "Female", "Male", "Female", "Male", "Female", "Male", "Female", "Male", "Female"]
}

df = pd.DataFrame(data)

df.to_csv("employees.csv", index=False)

print("employees.csv created successfully")
import pandas as pd

df = pd.read_csv("employees.csv")
df.head()

import pandas as pd
import sqlite3

df_csv = pd.read_csv("employees.csv")

df_csv.to_excel("employees.xlsx", index=False)
df_csv.to_json("employees.json", orient="records")

conn = sqlite3.connect("employees.db")
df_csv.to_sql("employees", conn, if_exists="replace", index=False)

df_excel = pd.read_excel("employees.xlsx")
df_json = pd.read_json("employees.json")
df_sql = pd.read_sql("SELECT * FROM employees", conn)

print("read_csv()")
print(df_csv)

print("\nread_excel()")
print(df_excel)

print("\nread_json()")
print(df_json)

print("\nread_sql()")
print(df_sql)

conn.close()

import pandas as pd

df = pd.read_csv("employees.csv")

print("head()")
print(df.head())

print("\ntail()")
print(df.tail())

print("\ninfo()") #summary, no.of rows, columns, column names, no. of non-null values, data type, memory usage
df.info()

print("\ndescribe()") #statistical summary: Count,mean,(std), min value, 25%, 50% (median), and 75% percentiles,maximum value

print(df.describe())

print("\nshape")
print(df.shape)

print("\ncolumns") #names of columns
print(df.columns)

print("\nindex") #means rows are numbered 1 to 11, returns row labels
print(df.index)

print("\ndtypes") #data type of each column
print(df.dtypes)

import pandas as pd

df = pd.read_csv("employees.csv")

print("loc[]") #df.loc[0:3] Returns rows with labels 0,1,2,3, loc includes the ending index. specific columns df.loc[:, ["Name", "Salary"]]
print(df.loc[2])

print("\niloc[]") #select data using integer positions, iloc excludes the ending index.
print(df.iloc[2])

print("\nat[]") #retrieve a single value
print(df.at[2, "Name"])

print("\niat[]")
print(df.iat[2, 1])

import pandas as pd #detect, remove, or replace missing values

df = pd.read_csv("employees.csv")

print("isnull()")
print(df.isnull())

print("\nnotnull()")
print(df.notnull())

print("\ndropna()") #removes rows (or columns) containing missing values.
print(df.dropna())

print("\nfillna()")
print(df.fillna(0))

print("\nreplace()")
print(df.replace("Mumbai", "Navi Mumbai"))

import pandas as pd

df = pd.read_csv("employees.csv")

print("drop()") #remove rows or columns
print(df.drop(columns=["Gender"]))

print("\nrename()") #change row or column names.
print(df.rename(columns={"Salary": "Income"}))

print("\nsort_values()") #sort based on column values
print(df.sort_values(by="Salary"))

print("\nsort_index()") #sort acc to row index
print(df.sort_index(ascending=False))

print("\ngroupby()")
print(df.groupby("Department")["Salary"].mean())

dept = pd.DataFrame({
    "Department": ["IT", "HR", "Sales", "Finance", "Marketing"],
    "Manager": ["Akshat", "Pranjal", "Ajay", "Taniya", "Kirti"]
})

print("\nmerge()") #based on common column
print(pd.merge(df, dept, on="Department"))

df1 = df.set_index("Department")
df2 = dept.set_index("Department")

print("\njoin()") #based on index
print(df1.join(df2))

print("\nconcat()")
print(pd.concat([df.head(3), df.tail(3)]))

print("\npivot()") #reshapes, Each department becomes a new column, index,column combinations should be unique otherwise error
print(df.pivot(index="ID", columns="Department", values="Salary"))

print("\npivot_table()") #can handle duplicate values by applying an aggregation function.
print(df.pivot_table(values="Salary", index="Department", aggfunc="mean"))

print("\nmelt()") #converts wide format data into long format. variable, value
print(pd.melt(df, id_vars=["ID", "Name"], value_vars=["Age", "Salary"]))

import pandas as pd

df = pd.read_csv("employees.csv")

print("mean()")
print(df.mean(numeric_only=True))

print("\nmedian()")
print(df.median(numeric_only=True))

print("\nmode()")
print(df.mode())

print("\nstd()")
print(df.std(numeric_only=True))

print("\nvar()")
print(df.var(numeric_only=True))

print("\ncorr()")
print(df.corr(numeric_only=True))

print("\ncov()")
print(df.cov(numeric_only=True))

print("\nvalue_counts()")
print(df["Department"].value_counts())

print("\nunique()")
print(df["Department"].unique())

print("\nnunique()")
print(df["Department"].nunique())

import matplotlib.pyplot as plt
import numpy as np

x = [1, 2, 3, 4, 5]
y = [10, 20, 15, 25, 30]

plt.figure(figsize=(12, 8))

plt.subplot(3, 3, 1)
plt.plot(x, y, label="Line")
plt.title("Plot")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)

plt.subplot(3, 3, 2)
plt.scatter(x, y)
plt.title("Scatter")

plt.subplot(3, 3, 3)
plt.bar(x, y)
plt.title("Bar")

plt.subplot(3, 3, 4)
plt.barh(x, y)
plt.title("Horizontal Bar")

plt.subplot(3, 3, 5)
plt.hist(y)
plt.title("Histogram") #freq distribution

plt.subplot(3, 3, 6)
plt.pie(y, labels=x, autopct="%1.1f%%")
plt.title("Pie")

plt.subplot(3, 3, 7) #min value, 25th tile, median, 75th, max value
plt.boxplot(y)
plt.title("Boxplot")

plt.subplot(3, 3, 8) #median, spread, density
plt.violinplot(y)
plt.title("Violin Plot")

image = np.array([[10, 20, 30],
                  [40, 50, 60],
                  [70, 80, 90]])

plt.subplot(3, 3, 9)
plt.imshow(image)
plt.title("Image")

plt.tight_layout()

plt.savefig("graphs.png")

plt.show()

from sklearn.datasets import load_iris, load_digits, fetch_california_housing
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

iris = load_iris() #The dataset has 150 samples, 4 features, and 3 flower species. .data contains feature values while .target contains class labels.
print("load_iris()")
print(iris.data[:5])
print(iris.target[:5])

digits = load_digits() #Each row = one image, Each column = pixel intensity, Total columns = 64
print("\nload_digits()")
print(digits.data[:5])
print(digits.target[:5]) #Every image is converted into 64 numerical features representing pixel intensities. The target contains the actual digit (0–9).

housing = fetch_california_housing()
print("\nfetch_california_housing()")
print(housing.data[:5]) #contains feature values
print(housing.target[:5]) #contains house prices

X1, y1 = make_classification(
    n_samples=100,
    n_features=4,
    n_classes=2,
    random_state=42
) #synthetic data for classification, 2 classes:0 or 1

print("\nmake_classification()")
print(X1[:5])
print(y1[:5])

X2, y2 = make_regression(
    n_samples=100,
    n_features=3,
    random_state=42
) #continuous numerical target values instead of class labels.

print("\nmake_regression()")
print(X2[:5])
print(y2[:5])

X_train, X_test, y_train, y_test = train_test_split(
    X1,
    y1,
    test_size=0.2,
    random_state=42
)

print("\ntrain_test_split()")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import Normalizer, LabelEncoder
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv("employees.csv")

print("StandardScaler()")
scaler = StandardScaler()
print(scaler.fit_transform(df[["Age", "Salary"]].fillna(0)))

print("\nMinMaxScaler()")
minmax = MinMaxScaler()
print(minmax.fit_transform(df[["Age", "Salary"]].fillna(0)))

print("\nRobustScaler()")
robust = RobustScaler()
print(robust.fit_transform(df[["Age", "Salary"]].fillna(0)))

print("\nNormalizer()")
normalizer = Normalizer()
print(normalizer.fit_transform(df[["Age", "Salary"]].fillna(0)))

print("\nLabelEncoder()")
label = LabelEncoder()
print(label.fit_transform(df["Department"]))

print("\nOneHotEncoder()")
onehot = OneHotEncoder(sparse_output=False)
print(onehot.fit_transform(df[["Department"]]))

print("\nOrdinalEncoder()")
ordinal = OrdinalEncoder()
print(ordinal.fit_transform(df[["Department"]]))

print("\nPolynomialFeatures()")
poly = PolynomialFeatures(degree=2)
print(poly.fit_transform(df[["Experience"]]))

#LabelEncoder() – Converts categories into integer labels.
#OneHotEncoder() – Converts categories into binary (0/1) columns.
#OrdinalEncoder() – Converts ordered categories into integer values.
#PolynomialFeatures() – Creates polynomial features from existing features.

from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.model_selection import RandomizedSearchCV, KFold, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
X = iris.data
y = iris.target

model = DecisionTreeClassifier(random_state=42)

print("cross_val_score()") #Evaluates a model using cross-validation and returns the score for each fold.
scores = cross_val_score(model, X, y, cv=5)
print(scores)

print("\nKFold()") #Splits the dataset into K equal folds for cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kf)
print(scores)

print("\nStratifiedKFold()") #Splits the dataset into K folds while preserving class distribution.
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skf)
print(scores)

params = {
    "max_depth": [2, 3, 4, 5],
    "criterion": ["gini", "entropy"]
}

print("\nGridSearchCV()") #Finds the best hyperparameters by testing all possible combinations.
grid = GridSearchCV(model, params, cv=5)
grid.fit(X, y)
print(grid.best_params_)
print(grid.best_score_)

print("\nRandomizedSearchCV()") #random parameter combinations
random = RandomizedSearchCV(model, params, n_iter=4, cv=5, random_state=42)
random.fit(X, y)
print(random.best_params_)
print(random.best_score_)