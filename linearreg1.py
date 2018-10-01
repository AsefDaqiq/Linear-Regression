import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset = pd.read_csv('Salary_Data.csv')

x = dataset.iloc[:,:-1]
y = dataset.iloc[:,1]


#spliting the data set into train and test 
from sklearn import cross_validation
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = cross_validation.train_test_split(x,y, test_size = 1/3, random_state = 0)

#Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train, y_train)

# predicting the Test set result

y_pred = regressor.predict(x_test)
 
#visualising the training set result
x_train.sort()

plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title('salary vs experience')
plt.xlabel('expreinece')
plt.ylabel('salary')
plt.show()

