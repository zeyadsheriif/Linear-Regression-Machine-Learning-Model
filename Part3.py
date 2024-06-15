!pip install -U scikit-learn

# Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import SGDRegressor

# Load Iris dataset
iris = datasets.load_iris()
iris

# use the numpy concatenate function
iris_df = pd.DataFrame(data = np.c_[iris['data'],iris['target']],
                       columns = iris['feature_names'] + ['target'])

# Show the dataset infromation
iris_df.info()

#Show head of dataset
print(iris_df.head())

#Describe tahe dataset
iris_df.describe()

#check the samples for each class / is it balanced dataset
iris_df.target.value_counts().plot(kind= 'bar');

iris_df.target.value_counts()

#check for missing data
print('missing values -> {}'.format (iris_df.isna().sum()))  # -> why ??

#check duplicates
print('dubblicate values -> {}'.format (iris_df.duplicated()))

#drop duplicates
iris_df.drop_duplicates(inplace = True)
#test after remove the duplicates
print(iris_df.duplicated().sum())

iris_df['Augmentation'] = np.ones(149)
print(iris_df)
b = iris_df['Augmentation']
print(b)

X = iris_df.iloc[:,:-1]
y = iris_df.iloc[:,:-1]

X_train, X_test,y_train, y_test = train_test_split(X,y, test_size=0.2, shuffle=True, random_state=0)
X_train = np.asarray(X_train)
y_train = np.asarray(y_train)
X_test = np.asarray(X_test)
y_test = np.asarray(y_test)

#check the traing set size and test set size:
print("Training set size:", len(X_train), "samples")
print("Test set size:", len(X_test), "samples")

#check the traing set shape and test set shape:
print("Training set shape:", X_train.shape, "samples")
print("Test set shape:", X_test.shape, "samples")

print("Shape of y_train:", y_train.shape)

y1_setosa = np.where(np.any(y_train == 0, axis=1), 1, -1)
print("Shape of y1_setosa:", y1_setosa.shape)
y1_setosa = y1_setosa.reshape(len(y1_setosa))
y1_setosa

y2_versicolor = np.where(np.any(y_train == 1, axis=1), 1, -1)
print("Shape of y3_virginica:", y2_versicolor.shape)
y2_versicolor = y2_versicolor.reshape(len(y2_versicolor))
y2_versicolor

y3_virginica = np.where(np.any(y_train == 2, axis=1), 1, -1)
print("Shape of y3_virginica:", y3_virginica.shape)
y3_virginica = y3_virginica.reshape(len(y3_virginica))
y3_virginica

def pseudoinverse(X):
  X_tr = X.transpose()
  X_ps = np.linalg.inv(X_tr.dot(X)).dot(X_tr)
  return X_ps

w1 = pseudoinverse(X_train).dot(y1_setosa)
w1

w2 = pseudoinverse(X_train).dot(y2_versicolor)
w2

w3 = pseudoinverse(X_train).dot(y3_virginica)
w3

#Equations
y1_setosa = np.array(X_test.dot(w1))
y2_versicolor = np.array(X_test.dot(w2))
y3_virginica = np.array(X_test.dot(w3))

y1_setosa

y2_versicolor

y3_virginica

def test_model(X, w1, w2, w3):
    y1_setosa = np.array(X.dot(w1))
    y2_versicolor = np.array(X.dot(w2))
    y3_virginica = np.array(X.dot(w3))
    y = []

    for i in range(len(y1_setosa)):
        class1 = y1_setosa[i] > 0 and y2_versicolor[i] < 0 and y3_virginica[i] < 0
        class2 = y1_setosa[i] < 0 and y2_versicolor[i] > 0 and y3_virginica[i] < 0
        class3 = y1_setosa[i] > 0 and y2_versicolor[i] < 0 and y3_virginica[i] > 0

        undefined = not (class1 ^ class2 ^ class3)  # XNOR operator

        if class1:
            y.append('setosa')
        if class2:
            y.append('versicolor')
        if class3:
            y.append('virginica')
        if undefined:
            y.append('undefined')

    return np.array(y)

prediction = test_model(X_test, w1, w2, w3)
print(prediction)
