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

# Convert iris dataset to DataFrame
iris_df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                       columns=iris['feature_names'] + ['target'])

# Group samples of class 2 and class 3 together to form Class II
iris_df['target'] = iris_df['target'].map({0: 1, 1: -1, 2: -1})

# Show the dataset infromation
iris_df.info()

#Show head of dataset
print(iris_df.head())

#Describe the dataset
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

# Splitting Class I samples
class_i_data = iris_df[iris_df['target'] == 1]
class_i_train = class_i_data.sample(n=40, random_state=42)
class_i_test = class_i_data.drop(class_i_train.index)

#check the traing set size and test set size:
print("Training set size:", len(class_i_train), "samples")
print("Test set size:", len(class_i_test), "samples")

#check the traing set shape and test set shape:
print("Training set shape:", class_i_train.shape, "samples")
print("Test set shape:", class_i_test.shape, "samples")

# Splitting Class II samples
class_ii_data = iris_df[iris_df['target'] == -1]
class_ii_train = class_ii_data.sample(n=80, random_state=42)
class_ii_test = class_ii_data.drop(class_ii_train.index)

#check the traing set size and test set size:
print("Training set size:", len(class_ii_train), "samples")
print("Test set size:", len(class_ii_test), "samples")

#check the traing set shape and test set shape:
print("Training set shape:", class_ii_train.shape, "samples")
print("Test set shape:", class_ii_test.shape, "samples")

# Combining train and test sets
train_data = pd.concat([class_i_train, class_ii_train])
test_data = pd.concat([class_i_test, class_ii_test])

# Separate features and target for training and testing
X_train = train_data.drop(columns=['target'])
y_train = train_data['target']
X_test = test_data.drop(columns=['target'])
y_test = test_data['target']

# Define pseudoinverse function
def pseudoinverse(X):
    X_tr = X.transpose()
    X_ps = np.linalg.inv(X_tr.dot(X)).dot(X_tr)
    return X_ps

# Define function to calculate weights
def calculate_weights(X_train, y_train):
    X_ps = pseudoinverse(X_train)
    return X_ps.dot(y_train)

# Calculate weights for Class I
y1_setosa = np.where(y_train == 1, 1, -1)
print("Shape of y1_setosa:", y1_setosa.shape)
y1_setosa = y1_setosa.reshape(len(y1_setosa))
y1_setosa

w1 = calculate_weights(X_train, y1_setosa)
w1

# Calculate weights for Class II
y2_versicolor_virginica = np.where(y_train != 1, 1, -1)
print("Shape of y2_versicolor_virginica:", y2_versicolor_virginica.shape)
y2_versicolor_virginica = y2_versicolor_virginica.reshape(len(y2_versicolor_virginica))
y2_versicolor_virginica

w2 = calculate_weights(X_train, y2_versicolor_virginica)

# Equations
y1_setosa_pred = np.array(X_test.dot(w1))
y2_versicolor_virginica_pred = np.array(X_test.dot(w2))

y1_setosa_pred
y2_versicolor_virginica_pred

# Define function to classify test samples
def classify_samples(y1_setosa_pred, y2_versicolor_virginica_pred):
    y_pred = []
    for i in range(len(y1_setosa_pred)):
        if y1_setosa_pred[i] > 0:
            y_pred.append(1)
        else:
            if y2_versicolor_virginica_pred[i] > 0:
                y_pred.append(-1)
            else:
                y_pred.append(-1)  # In case of tie, classify as Class II
    return y_pred

# Classify test samples
y_pred = classify_samples(y1_setosa_pred, y2_versicolor_virginica_pred)
print(y_pred)

# Define function to classify test samples
def classify_samples(y1_setosa_pred, y2_versicolor_virginica_pred):
    y_pred = []
    for i in range(len(y1_setosa_pred)):
        if y1_setosa_pred[i] > 0:
            y_pred.append("Class I")
        else:
            if y2_versicolor_virginica_pred[i] > 0:
                y_pred.append("Class II")
            else:
                y_pred.append("Class II")  # In case of tie, classify as Class II
    return y_pred

# Classify test samples
y_pred = classify_samples(y1_setosa_pred, y2_versicolor_virginica_pred)
print(y_pred)














