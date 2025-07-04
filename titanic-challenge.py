# titanic_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Load data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Combine for preprocessing
data = pd.concat([train, test], sort=False)

# === Data Cleaning ===

# Fill missing Age with median
# data["Age"].fillna(data["Age"].median(), inplace=True)
data["Age"] = data["Age"].fillna(data["Age"].median())

# Fill missing Embarked with mode
# data["Embarked"].fillna(data["Embarked"].mode()[0], inplace=True)
data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode()[0])

# Fill missing Fare with median
# data["Fare"].fillna(data["Fare"].median(), inplace=True)
data["Fare"] = data["Fare"].fillna(data["Fare"].median())

# Drop Cabin due to too many missing values
data.drop("Cabin", axis=1, inplace=True)

# === Feature Engineering ===

# Encode Sex
data["Sex"] = data["Sex"].map({"male": 0, "female": 1})

# Encode Embarked
# data["Embarked"] = data["Embarked"].map({"S": 0, "C": 1, "Q": 2})
data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode()[0])
data["Embarked"] = data["Embarked"].map({"S": 0, "C": 1, "Q": 2})


# Family size and is alone
data["FamilySize"] = data["SibSp"] + data["Parch"] + 1
data["IsAlone"] = (data["FamilySize"] == 1).astype(int)

# Extract Title from Name
# data["Title"] = data["Name"].str.extract(" ([A-Za-z]+)\.", expand=False)
data["Title"] = data["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
title_mapping = {
    "Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs", "Master": "Master",
    "Dr": "Rare", "Rev": "Rare", "Col": "Rare", "Major": "Rare",
    "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs", "Countess": "Rare",
    "Lady": "Rare", "Jonkheer": "Rare", "Don": "Rare", "Dona": "Rare",
    "Capt": "Rare", "Sir": "Rare"
}
data["Title"] = data["Title"].map(title_mapping)
data["Title"] = data["Title"].fillna("Rare")

# One-hot encode titles
title_dummies = pd.get_dummies(data["Title"], prefix="Title")
data = pd.concat([data, title_dummies], axis=1)

# Age bins
data["AgeBin"] = pd.cut(data["Age"], bins=[0, 12, 20, 40, 60, 80], labels=False)

# === Final Dataset Preparation ===

# Drop unused columns
drop_cols = ["PassengerId", "Name", "Ticket", "Title", "Age", "Fare"]
data.drop(columns=drop_cols, inplace=True)

# Separate back into train and test
X_train = data[:len(train)]
X_test = data[len(train):]
y_train = train["Survived"]

# === Modeling ===

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr_score = cross_val_score(lr, X_train, y_train, cv=5).mean()
print(f"Logistic Regression CV score: {lr_score:.4f}")

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_score = cross_val_score(rf, X_train, y_train, cv=5).mean()
print(f"Random Forest CV score: {rf_score:.4f}")

# Train final model and predict
rf.fit(X_train, y_train)
predictions = rf.predict(X_test)

# === Submission ===

submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": predictions
})
submission.to_csv("submission.csv", index=False)
print("Submission file saved as 'submission.csv'")
