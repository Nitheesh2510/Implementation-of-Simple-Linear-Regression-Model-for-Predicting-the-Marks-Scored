# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas.
## Program:
```

Program to implement the simple linear regression model for predicting the marks scored.
Developed by: NITHEESH KUMAR B
RegisterNumber:  212224230189

```
```

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
```
## Output:

<img width="243" height="812" alt="418852378-de8a194b-b1ab-4e5b-893b-ed833b3ace46" src="https://github.com/user-attachments/assets/57271156-d4cb-46c2-af34-f7894a1e3c8e" />


<img width="262" height="514" alt="418852464-aa7e9e93-bd28-4f11-9991-1eda9cd571db" src="https://github.com/user-attachments/assets/eaaa2945-ddcc-4dbc-9ad1-ddf5f9cbb9b1" />

<img width="951" height="819" alt="418852552-13a9c74e-d3ef-418f-b89b-f09c782018f0" src="https://github.com/user-attachments/assets/78d4ed7f-c7b2-4bc9-87e2-ec895471df21" />



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
