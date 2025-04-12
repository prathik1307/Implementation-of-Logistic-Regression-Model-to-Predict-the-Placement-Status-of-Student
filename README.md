# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
 1.Import the required packages and print the present data.
 2.Print the placement data and salary data. 
 3.Find the null and duplicate values.
 4.Using logistic regression find the predicted values of accuracy , confusion matrices. 
 5.Display the results.
 
``` 
 

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: PRATHIKSHA.R
RegisterNumber:212224040244

import pandas as pd
data=pd.read_csv(r"C:\Users\admin\Desktop\Placement_Data.csv")
data.head()
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()
data1.isnull().sum()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1
x=data1.iloc[:,:-1]
x
y=data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
print(y_pred)
print()
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
print(accuracy)
print()
from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
print(confusion)
print()
from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])


 
*/
```

## Output:
![the Logistic Regression Model to Predict the Placement Status of Student](sam.png)

HEAD:

![image](https://github.com/user-attachments/assets/707d1115-48c9-49bc-8fda-6b54cd5836c5)


COPY:

![image](https://github.com/user-attachments/assets/ee9e2487-419e-4a68-9018-9bf54194c090)


FIT TRANSFORM:

![image](https://github.com/user-attachments/assets/546e26af-4a8e-4e4a-9e88-a38de574ac11)


LOGISTIC REGREESION :

![image](https://github.com/user-attachments/assets/4d3ff3f6-e7bf-4785-a267-f768175e2e04)


ACCURACY SCORE:

![image](https://github.com/user-attachments/assets/6aae5311-0f16-4f35-abde-f6d4f64be24c)


CONFUSION MATRIX:

![image](https://github.com/user-attachments/assets/8fca4442-92d2-40bb-aae7-588eaecb5794)


CLASSFICATION REPORT:

![image](https://github.com/user-attachments/assets/7cd6dfde-af5c-4ddb-97f8-2ff211924bc8)


PREDICTION:

![image](https://github.com/user-attachments/assets/5b947290-9ac8-47cb-a462-105c5ca9fc0d)










## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
