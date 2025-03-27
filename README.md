# Bsics Titanic-Survival-Prediction
Developed a classification model to predict passenger survival in the Titanic disaster using supervised learning.

Step 1:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

Step 2: Load the Dataset from Kaggle -:  https://www.kaggle.com/c/titanic(Titanic - Machine Learning from Disaster)
df = pd.read_csv("/train.csv")
print(df.head())

Step 3: Data Cleaning - control Missing values

df["Age"].fillna(df["Age"].median(), inplace=True)
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

Step 4: Convert Categorical to Numerical

df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df["Embarked"] = df["Embarked"].map({"C": 0, "Q": 1, "S": 2})

Step 5: Create new feature 

df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

Step 6: Slect feature for tarining 

features = ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize"]
X = df[features]
y = df["Survived"]

Step 7: Train test split 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

Step 8:Train test logistic regression model

model = LogisticRegression()
model.fit(X_train, y_train)

Step 9:Prediciton

y_pred = model.predict(X_test)

Step 10: To Evalution the model

accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

Confusion Matrix and Classification 

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

Step 11:

Viusualizde Survival By gender

plt.figure(figsize=(6,4))
sns.barplot(x="Sex", y="Survived", data=df)
plt.xticks(ticks=[0, 1], labels=["Male", "Female"])
plt.title("Survival Rate by Gender")
plt.show()



