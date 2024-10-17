# EX2 Implementation of Simple Linear Regression Model for Predicting the Marks Scored
## DATE:
## AIM:
To implement simple linear regression using sklearn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Get the independent variable X and dependent variable Y by reading the dataset.
2. Split the data into training and test data.
3. Import the linear regression and fit the model with the training data.
4. Perform the prediction on the test data.
5. Display the slop and intercept values.
6. Plot the regression line using scatterplot.
7. Calculate the MSE.

## Program:
```
/*
Program to implement univariate Linear Regression to fit a straight line using least squares.
Developed by: kavipriya s.p
RegisterNumber: 2305002011 
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('/content/ex1.csv')
df.head()
df.tail()
X=df.iloc[:,:-1].values
X
Y=df.iloc[:,1].values
Y
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
Y_pred
Y_test
plt.scatter(X_train,Y_train,color="pink")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(X_test,Y_test,color="purple")
plt.plot(X_train,regressor.predict(X_train),color="yellow")
plt.title("Hours vs scores(Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:

![Screenshot 2024-10-17 100208](https://github.com/user-attachments/assets/034318a6-7c32-41ae-ad9b-6ef13af0936e)


![Screenshot 2024-10-17 100254](https://github.com/user-attachments/assets/5636c138-eeab-4c50-bb63-0a2779d5a4e4)


![Screenshot 2024-10-17 100300](https://github.com/user-attachments/assets/abce3d6f-e397-4966-b6bf-949cca1b4867)


![Screenshot 2024-10-17 100307](https://github.com/user-attachments/assets/98ffa034-d384-45a6-b69c-a3e2460269e8)

![Screenshot 2024-10-17 100530](https://github.com/user-attachments/assets/eddebb91-5622-4edd-a318-501a8b5f9522)

![Screenshot 2024-10-17 100544](https://github.com/user-attachments/assets/7e95f678-a9e5-4a4c-a3d1-c80a93616d2a)

![Screenshot 2024-10-17 100606](https://github.com/user-attachments/assets/8a739176-aad1-43dc-a63f-79568d486ccb)

![Uploading Screenshot 2024-10-17 100624.png…]()

## Result:
Thus the univariate Linear Regression was implemented to fit a straight line using least squares using python programming.
