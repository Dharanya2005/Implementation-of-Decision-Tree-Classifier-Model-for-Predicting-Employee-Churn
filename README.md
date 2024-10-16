# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.import pandas module and import the required data set.

2.Find the null values and count them.

3.Count number of left values.

4.From sklearn import LabelEncoder to convert string values to numerical values.

5.From sklearn.model_selection import train_test_split.

6.Assign the train dataset and test dataset.

7.From sklearn.tree import DecisionTreeClassifier.

8.Use criteria as entropy.

9.From sklearn import metrics.

10.Find the accuracy of our model and predict the require values. 
## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: DHARANYA N
RegisterNumber:  212223230044
*/
```

## Output:
```
import pandas as pd
data=pd.read_csv("Employee.csv")
data.head()
```
![image](https://github.com/user-attachments/assets/5eed6e9c-c63a-4632-94b9-52828b072d42)
```
data.info()
```
![image](https://github.com/user-attachments/assets/3ba00b0a-29cf-4e96-b76f-6f9058f4605b)
```
data.isnull().sum()
```
![image](https://github.com/user-attachments/assets/89c6de15-3cc3-4301-8ec0-a48b4b174819)
```
data["left"].value_counts()
```
![image](https://github.com/user-attachments/assets/84f656ec-188b-4748-9bf2-9ae8b74eb8cc)
```
from sklearn.preprocessing import LabelEncoder 
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
```
![image](https://github.com/user-attachments/assets/d269a43d-8f29-413a-a71c-631bbafd87d4)
```
x=data[["satisfaction_level", "last_evaluation", "number_project","average_montly_hours", "time_spend_company", "Work_accident", "promotion_last_5years", "salary"]]

x.head()
```
![image](https://github.com/user-attachments/assets/0100c66f-d2c1-4796-a835-8e2133f1f56f)
```
y=data["left"]
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test=train_test_split(x, y, test_size=0.2, random_state=100)
from sklearn.tree import DecisionTreeClassifier 
dt=DecisionTreeClassifier (criterion="entropy") 
dt.fit(x_train, y_train) 
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```
![image](https://github.com/user-attachments/assets/37b5e531-db53-43a4-85d9-e9bc4f5c61f0)
```
dt.predict([[0.5,0.8,9,260,6,0,1,]])
```
![image](https://github.com/user-attachments/assets/d8b00ef8-e2e9-439e-af4c-a25ca9e34958)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
