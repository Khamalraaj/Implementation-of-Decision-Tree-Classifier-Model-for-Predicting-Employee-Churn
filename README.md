# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.

2. Upload and read the dataset.

3. Check for any null values using the isnull() function.

4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.

5. Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Khamalraaj S
RegisterNumber:  212224230122
*/
```
```py
import pandas as pd
data = pd.read_csv("Employee (1).csv")
data.head()

data.info()
data.isnull().sum()
data['left'].value_counts()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

data['salary'] = le.fit_transform(data['salary'])
data.head()

x=data[['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','salary']]
x.head()

y=data['left']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state =100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion='entropy')
dt.fit(x_train,y_train)
y_predict=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_predict)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:

### DATA HEAD:
<img width="1379" height="253" alt="image" src="https://github.com/user-attachments/assets/0b9948c5-5be8-4cb9-ba3b-01dd11ebaa1d" />


<br>
<br>
<br>


### DATASET INFO:
<img width="686" height="422" alt="image" src="https://github.com/user-attachments/assets/41565fad-330c-4bb2-ac8d-e1c92c2b7fbe" />


### NULL DATASET:
<img width="300" height="114" alt="image" src="https://github.com/user-attachments/assets/52860583-173c-47e8-aa09-0dc2e6164b20" />


### VALUES COUNT IN LEFT COLUMN:
<img width="335" height="278" alt="image" src="https://github.com/user-attachments/assets/83c85dbf-9448-4843-b4a7-e49736704349" />


### DATASET TRANSFORMED HEAD:
<img width="1370" height="276" alt="image" src="https://github.com/user-attachments/assets/8f92eedc-eec1-49d6-92d2-726ab48debd3" />


### X.HEAD:
<img width="1264" height="248" alt="image" src="https://github.com/user-attachments/assets/afd91e0a-cf10-4898-a5ac-e71605eea861" />


### ACCURACY:
<img width="227" height="49" alt="image" src="https://github.com/user-attachments/assets/be43f50a-3fe4-4789-9b7d-42b47e9673b0" />

### DATA PREDICTION:

<img width="285" height="60" alt="image" src="https://github.com/user-attachments/assets/185d3009-1ab8-4347-8981-53f36129db07" />


<br>



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
