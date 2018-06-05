# Machine_learning_python

Machine Learning in Python
==========================

## Getting Started ##

There are some classic machine learning examples available online worth examining more closely
Namely, one called the Boston Housing Machine learning problem.

This is a Linear Regression problem that uses a dataset of housing prices in Boston. The dataset also includes additional features like the Crime rate and age of the home owners.

### How do you begin? ###

Python code usually begins by importing libraries that are required in order to able to use some functionality in code.

```
import pandas as pd
```
Pandas is an open-source library for data analysis tools
```
import numpy as np
```
Numpy is an open-source scientific computing tool for Python
```
from sklearn import preprocessing
```
This command pre-processes the data, making it more standardized and easier to use for scikit-learn
```
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
```
Matplotlib is a plotting tool, plt is what we are calling the reference in the example
```
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
```
These are tools necessary for the logistic regression we will be doing with the data
```
import seaborn as sns

sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
```
Seaborn is used for statistical data visualization

## Now that we have the libraries and tools set, the next step would be to access the data ##

The Boston dataset can be imported
```
from sklearn.datasets import load_boston
boston = load_boston()
```

If you want to see what the headers of the columns are for this dataset, aka the keys, use the command:
```
print(boston.keys())
```

If you want to see the dimensions of the dataset, use the command:
```
print(boston.data.shape)
```

##The next step: Processing the data##
```
X = boston.drop('PRICE', axis = 1)
Y = boston['PRICE']
```

Training and testing the data using scikit-learn cross validation:
```
X_train, X_test, Y_train, Y_test = sklearn.cross_validation.train_test_split(X, Y, test_size = 0.33, random_state = 5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
```

### Linear Regression ###
```
from sklearn.linear_model import LinearRegression

lm = LinearRegression()
lm.fit(X_train, Y_train)
```

Calculate the prediction
```
Y_pred = lm.predict(X_test)

plt.scatter(Y_test, Y_pred)
plt.xlabel("Prices: $Y_i$")
plt.ylabel("Predicted prices: $\hat{Y}_i$")
plt.title("Prices vs Predicted prices: $Y_i$ vs $\hat{Y}_i$")
```


Finding the mean-square error
```
mse = sklearn.metrics.mean_squared_error(Y_test, Y_pred)
print(mse)
```
