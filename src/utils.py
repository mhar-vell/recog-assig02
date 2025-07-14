# utils.py

def load_data(file_path):
    import pandas as pd
    return pd.read_csv(file_path)

def preprocess_data(data):
    # Fill missing values
    data["Age"].fillna(data["Age"].median(), inplace=True)
    data["Embarked"].fillna(data["Embarked"].mode()[0], inplace=True)
    data["Fare"].fillna(data["Fare"].median(), inplace=True)
    
    # Encode categorical variables
    data["Sex"] = data["Sex"].map({"male": 0, "female": 1})
    data["Embarked"] = data["Embarked"].map({"S": 0, "C": 1, "Q": 2})
    
    return data

def feature_engineering(data):
    # Create additional features if necessary
    data["FamilySize"] = data["SibSp"] + data["Parch"] + 1
    data["IsAlone"] = (data["FamilySize"] == 1).astype(int)
    return data

def evaluate_model(y_true, y_pred):
    from sklearn.metrics import classification_report, confusion_matrix
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred))