# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1. Data Preparation: Load the Iris dataset, create a DataFrame, and split it into features (X) and target (y).

2. Data Splitting: Split the data into training and testing sets with an 80-20 ratio using train_test_split.

3. Model Training: Initialize an SGDClassifier and train it on the training data (X_train, y_train).

4. Prediction and Evaluation: Predict on the test set, calculate accuracy, and generate a confusion matrix for evaluation.


## Program and Output:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: SETHUKKARASI C
RegisterNumber:  212223230201
*/
```
<br>

```
# Import libraries.
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
```
<br>

```
# Load the Iris dataset
iris = load_iris()
```
<br>

```
# Create a Pandas DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
```
<br>

```
# Display the first few rows of the dataset
print(df.head())
```
<br>

![output1](/1.png)
<br>

```
# Split the data into features (X) and target (y)
X = df.drop('target', axis=1)
y = df['target']
```
<br>

```
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
<br>

```
# Create an SGD classifier with default parameters
sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3)
```
<br>

```
# Train the classifier on the training data
sgd_clf.fit(X_train, y_train)
```
<br>

![output2](/2.png)
<br>

```
# Make predictions on the testing data
y_pred = sgd_clf.predict(X_test)
```
<br>

```
# Evaluate the classifier's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")
```
<br>

![output3](/3.png)
<br>

```
# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
```
<br>

![output4](/4.png)
<br>

## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
