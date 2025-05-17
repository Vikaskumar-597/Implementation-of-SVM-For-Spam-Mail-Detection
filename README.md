# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the spam dataset and handle encoding properly.
2. Display basic information and check for null values.
3. Extract the message text as features (x) and labels (y) for classification.
4. Split the dataset into training and testing sets.
5. Convert the text data into numerical vectors using CountVectorizer.
6. Train an SVM classifier on the transformed training data.
7. Predict on test data and evaluate model accuracy using accuracy_score. 

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by  : VIKASKUMAR M 
RegisterNumber: 212224220122
*/
```
```
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
data = pd.read_csv("/content/spam.csv", encoding="Windows-1252")
data.head()
```
```
data.info(0)
```
```
data.isnull().sum() 
```
```
# separating the features and labels
x = data["v2"].values  # text messages
y = data["v1"].values  # labels: spam or ham
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
```
```
cv = CountVectorizer()
x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)
```
```
svc = SVC()
svc.fit(x_train, y_train)
```
```
y_pred = svc.predict(x_test)
y_pred
```
```
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy*100}%")
```

## Output:

**Head Values**

![image](https://github.com/user-attachments/assets/5f83ea9d-3d24-4804-84bc-9ccbd2788d71)

**Dataframe Info**

![image](https://github.com/user-attachments/assets/0a73125e-87aa-418d-92fc-17570c3313a0)

**Sum - Null Values**

![image](https://github.com/user-attachments/assets/f416e464-1332-4830-809d-272584d3ac6e)

**Training the model**

![image](https://github.com/user-attachments/assets/38e58f25-541d-4931-9639-43854bb024a6)

**Predicting the test data**

![image](https://github.com/user-attachments/assets/21988b99-0870-4e42-a190-47cbd9f7cc72)

**Accuracy**

![image](https://github.com/user-attachments/assets/682d3b7c-df9d-404b-9fe4-3631b7edd8fe)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
