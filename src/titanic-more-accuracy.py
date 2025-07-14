# IMPORTS AND SETTINGS
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb  # Install with: pip install xgboost
import lightgbm as lgb

import os
print(os.getcwd())

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV
from sklearn.ensemble import StackingClassifier, VotingClassifier


# DATA LOADING
# train = pd.read_csv("train.csv")
train = pd.read_csv("/Users/marcoreis/Insync/marco.a.reis@gmail.com/Google Drive/dutfpr/recognition/assignments/recog-assig02/train.csv")
# test = pd.read_csv("test.csv")
test = pd.read_csv("/Users/marcoreis/Insync/marco.a.reis@gmail.com/Google Drive/dutfpr/recognition/assignments/recog-assig02/test.csv")
# gender_submission = pd.read_csv("gender_submission.csv")
gender_submission = pd.read_csv("/Users/marcoreis/Insync/marco.a.reis@gmail.com/Google Drive/dutfpr/recognition/assignments/recog-assig02/gender_submission.csv")
data = pd.concat([train, test], sort=False)

# MISSING VALUES HANDLING
# Fill missing values for Age, Embarked, and Fare
data["Age"] = data["Age"].fillna(data["Age"].median())
data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode()[0])
data["Fare"] = data["Fare"].fillna(data["Fare"].median())

# Drop Cabin due to too many missing values
data.drop("Cabin", axis=1, inplace=True)

# FEATURE ENGINEERING
# Fill missing Age and Sex values before binning
data["Age"] = data["Age"].fillna(data["Age"].median())
data["Sex"] = data["Sex"].fillna(data["Sex"].mode()[0])

# Encode categorical variables
data["Sex"] = data["Sex"].map({"male": 0, "female": 1})
data["Embarked"] = data["Embarked"].map({"S": 0, "C": 1, "Q": 2})

# Family size and is alone
data["FamilySize"] = data["SibSp"] + data["Parch"] + 1
data["IsAlone"] = (data["FamilySize"] == 1).astype(int)

# Extract Title from Name and map to common titles
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

# Create Age categories
data["AgeGroup"] = pd.cut(
    data["Age"],
    bins=[0, 12, 20, 40, 60, 80],
    labels=["Child", "Teen", "Adult", "MiddleAged", "Senior"]
)

# One-hot encode AgeGroup
agegroup_dummies = pd.get_dummies(data["AgeGroup"], prefix="AgeGroup")
data = pd.concat([data, agegroup_dummies], axis=1)

# Create a binary feature for 'Countess' in the Name (special feature)
data["IsCountess"] = data["Name"].str.contains("Countess", case=False, na=False).astype(int)

# Age bins 
data["AgeBin"] = pd.cut(data["Age"], bins=[0, 12, 20, 40, 60, 80], labels=False)

# Save PassengerId for submission
test_passenger_ids = test["PassengerId"]


# ------------------------------------------------------------------
# One-hot encode Pclass
pclass_dummies = pd.get_dummies(data["Pclass"], prefix="Pclass")
data = pd.concat([data, pclass_dummies], axis=1)

# Interaction feature: Age * Pclass
data["AgeClass"] = data["Age"] * data["Pclass"]

# Interaction feature: Sex * Pclass
data["SexPclass"] = data["Sex"] * data["Pclass"]

# Median fare per Pclass
data["PclassMedianFare"] = data.groupby("Pclass")["Fare"].transform("median")


# DROP UNUSED COLUMNS
# Drop columns only after feature engineeringprint(data.columns)
data.drop([
    "SibSp", "Parch", "Name", "Ticket", "Title", "Age", "Fare", "PassengerId"
], axis=1, inplace=True)

print(data.columns)

# FINAL MISSING VALUES HANDLING
# Check for missing values
print(data.isnull().sum())
            
# for col in data.columns:
#     print(f"{col}: {data[col].dtype}")
#     if data[col].isnull().any():
#         # Numeric columns
#         if pd.api.types.is_numeric_dtype(data[col]):
#             data[col] = data[col].fillna(data[col].median())
#         # Boolean columns
#         elif pd.api.types.is_bool_dtype(data[col]):
#             data[col] = data[col].fillna(False)
#         # Categorical columns
#         elif pd.api.types.is_categorical_dtype(data[col]):
#             data[col] = data[col].fillna(data[col].mode()[0])
#         # Object columns (strings, mixed types)
#         elif pd.api.types.is_object_dtype(data[col]):
#             data[col] = data[col].fillna(data[col].mode()[0])
#         else:
#             # Fallback for any other dtype
#             data[col] = data[col].fillna(data[col].mode()[0])
print(data.info())

for col in data.columns:
    print(f"{col}: {data[col].dtype}")
    if data[col].isnull().any():
        # Numeric columns
        if pd.api.types.is_numeric_dtype(data[col]):
            data[col] = data[col].fillna(data[col].median())
        # Boolean columns
        elif pd.api.types.is_bool_dtype(data[col]):
            data[col] = data[col].fillna(False)
        # All other columns (object, category, etc.)
        else:
            data[col] = data[col].fillna(data[col].mode(dropna=True)[0])
            
# SPLIT BACK TO TRAIN AND TEST
# Now split into train and test
X_train = data[:len(train)]
X_test = data[len(train):]
y_train = train["Survived"]

# ENCODING FOR MODELING
# Encode categorical features for modeling
X_train_encoded = pd.get_dummies(X_train)
X_test_encoded = pd.get_dummies(X_test)

# Align columns so train and test have the same features
X_train_encoded, X_test_encoded = X_train_encoded.align(X_test_encoded, join='left', axis=1, fill_value=0)

# Check the data
print(X_train_encoded.info())
print(X_train_encoded.isnull().sum())
print(X_train_encoded.describe())

# EXPLORATORY DATA ANALYSIS
# ------ correlation matrix ------
plt.figure(figsize=(12,8))
# Only use numeric columns for correlation
numeric_cols = X_train_encoded.select_dtypes(include=[np.number])
sns.heatmap(numeric_cols.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation Matrix")
plt.show()

# ------ histograms of numeric features ------
# Only plot histograms for numeric columns
numeric_cols.hist(figsize=(15, 10))
plt.tight_layout()
plt.show()

# ------ feature importances using Random Forest ------
# Convert boolean columns to int and category columns to codes
X_train_rf = X_train_encoded.copy()
for col in X_train_rf.select_dtypes(include=["bool"]).columns:
    X_train_rf[col] = X_train_rf[col].astype(int)
for col in X_train_rf.select_dtypes(include=["category"]).columns:
    X_train_rf[col] = X_train_rf[col].cat.codes

# Family survival correlation
# This captures the idea that family members often survived or perished together
# Add code like this before trying to access FamilySize
train['FamilySize'] = train['SibSp'] + train['Parch'] + 1  # +1 for the passenger themselves
family_survival = {}
for grp, grp_df in train[['Survived', 'Name', 'FamilySize', 'PassengerId']].groupby(['FamilySize']):
    if (grp[0] > 1):
        for idx, row in grp_df.iterrows():
            family_survival[row['PassengerId']] = row['Survived']

# Apply family survival knowledge to both train and test
for df in [X_train_rf, X_test]:
    df['FamilySurvived'] = df.index.map(lambda x: family_survival.get(x, 0.5))

# Ticket frequency - passengers with same ticket had similar fate
ticket_counts = train['Ticket'].value_counts()
X_train_rf['TicketFrequency'] = train['Ticket'].map(ticket_counts)
X_test['TicketFrequency'] = test['Ticket'].map(ticket_counts)

# Position on ship (if you still have cabin info)
if 'Cabin' in train.columns:
    # Extract deck information
    X_train_rf['Deck'] = train['Cabin'].str[0].map({'A': 0, 'B': 0.4, 'C': 0.8, 'D': 1.2, 'E': 1.6, 'F': 2.0, 'G': 2.4, 'T': 2.8})
    X_test['Deck'] = test['Cabin'].str[0].map({'A': 0, 'B': 0.4, 'C': 0.8, 'D': 1.2, 'E': 1.6, 'F': 2.0, 'G': 2.4, 'T': 2.8})

# Fill NA values
X_train_rf = X_train_rf.fillna(0)

# For each categorical column in X_test, add the value '0' to its categories
for col in X_test.select_dtypes(include=['category']).columns:
    X_test[col] = X_test[col].cat.add_categories(0)

# Then fill NaN values
X_test_rf = X_test.fillna(0)



# Train a Random Forest model and plot feature importances
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_rf, y_train)
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names_plot = X_train_rf.columns

plt.figure(figsize=(10, 6))
plt.title("Feature Importances (Random Forest)")
plt.bar(range(X_train_rf.shape[1]), importances[indices])
plt.xticks(range(X_train_rf.shape[1]), feature_names_plot[indices], rotation=90)
plt.tight_layout()
plt.show()

# MODEL TRAINING AND EVALUATION
# ------ CROSS-VALIDATION
# Model comparison using cross-validation
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC()
}

cv_scores = {}
for name, model in models.items():
    score = cross_val_score(model, X_train_rf, y_train, cv=5).mean()
    cv_scores[name] = score
    print(f"{name} CV score: {score:.4f}")

plt.figure(figsize=(8,4))
plt.bar(cv_scores.keys(), cv_scores.values(), color='skyblue')
plt.ylabel("Mean CV Accuracy")
plt.title("Model Comparison (5-fold CV)")
plt.ylim(0, 1)
plt.show()

# ------ HYPERPARAMETER TUNING
# Using GridSearchCV for hyperparameter tuning
# Note: This is an example, you can adjust the parameters as needed

# Example: Random Forest hyperparameter tuning
rf_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10]
}
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=5, n_jobs=-1)
rf_grid.fit(X_train_rf, y_train)
print("Best RF params:", rf_grid.best_params_)
print("Best RF CV score:", rf_grid.best_score_)

# Example: Logistic Regression hyperparameter tuning
# logreg_params = {
#     'C': [0.01, 0.1, 1, 10, 100],
#     'solver': ['liblinear', 'lbfgs']
# }
# logreg_grid = GridSearchCV(LogisticRegression(max_iter=1000), logreg_params, cv=5, n_jobs=-1)
# logreg_grid.fit(X_train_rf, y_train)
# print("Best LogReg params:", logreg_grid.best_params_)
# print("Best LogReg CV score:", logreg_grid.best_score_)

# ------ FINAL MODEL TRAINING
# Predict on training set using encoded features
y_pred = rf.predict(X_train_rf)
cm = confusion_matrix(y_train, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix (Random Forest, Training Set)")
plt.show()

print(classification_report(y_train, y_pred))

# PREDICTION ON TEST SET
# ------ RANDOM FOREST PREDICTION
# Convert boolean and category columns in X_test to numeric types for prediction
X_test_rf = X_test_encoded.copy()
for col in X_test_rf.select_dtypes(include=["bool"]).columns:
    X_test_rf[col] = X_test_rf[col].astype(int)
for col in X_test_rf.select_dtypes(include=["category"]).columns:
    X_test_rf[col] = X_test_rf[col].cat.codes


# Train Random Forest on the encoded training set and create submission
# Convert boolean columns to int and category columns to codes
X_train_rf = X_train_encoded.copy()
for col in X_train_rf.select_dtypes(include=["bool"]).columns:
    X_train_rf[col] = X_train_rf[col].astype(int)
for col in X_train_rf.select_dtypes(include=["category"]).columns:
    X_train_rf[col] = X_train_rf[col].cat.codes
    
rf = RandomForestClassifier(n_estimators=100, max_depth= None, min_samples_split=2, random_state=42)
rf_score = cross_val_score(rf, X_train_rf, y_train, cv=5).mean()
print(f"Random Forest CV score: {rf_score:.4f}")
rf.fit(X_train_rf, y_train)
rf_predictions = rf.predict(X_train_rf)

# ------ LOGISTIC REGRESSION PREDICTION
# Train Logistic Regression on the encoded training set and create submission
# lr = LogisticRegression(max_iter=1000, C=0.1, solver='liblinear')
# lr_score = cross_val_score(lr, X_train_rf, y_train, cv=5).mean()
# print(f"Logistic Regression CV score: {lr_score:.4f}")
# lr.fit(X_train_encoded, y_train)
# lr_predictions = lr.predict(X_test_encoded)

# -----------------------------------------------------
# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth= None, min_samples_split=2, random_state=42)
rf.fit(X_train_rf, y_train)

# Get feature importances
importances = rf.feature_importances_
feature_names = X_train_rf.columns

# Sort features by importance
indices = np.argsort(importances)[::-1]
top_n = 15  # Number of top features to select

# Print top features
print("Top features:")
for i in range(top_n):
    print(f"{feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

# Select only top features for modeling
selected_features = feature_names[indices[:top_n]]
X_train_top = X_train_rf[selected_features]
X_test_top = X_test_rf[selected_features]

rf_top = RandomForestClassifier(n_estimators=300, max_features="sqrt", max_depth= None, min_samples_split=2, random_state=42)
rf_top.fit(X_train_top, y_train)
rf_top_predictions = rf_top.predict(X_test_top)

# >>>>> -----------------------------------
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'max_features': ['sqrt', 'log2']
}
grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, n_jobs=-1)
grid.fit(X_train_top, y_train)
print("Best params:", grid.best_params_)

# Use the best estimator for prediction
rf_top = grid.best_estimator_
rf_top_predictions = rf_top.predict(X_test_top)
# >>>>> -----------------------------------
# gbc = GradientBoostingClassifier(random_state=42)
# gbc.fit(X_train_top, y_train)
# gbc_predictions = gbc.predict(X_test_top)

# Compare your predictions with gender_submission
comparison = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "ModelPrediction": rf_top_predictions,
    # "ModelPrediction": gbc_predictions,
    "GenderSubmission": gender_submission["Survived"]
})

# Calculate accuracy
accuracy = (comparison["ModelPrediction"] == comparison["GenderSubmission"]).mean()
print(f"Agreement with gender_submission.csv: {accuracy:.4f}")

# ADVANCED MODEL OPTIMIZATION FOR ACCURACY
# from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
# from sklearn.svm import SVC
# from sklearn.preprocessing import StandardScaler
# import xgboost as xgb  # Install with: pip install xgboost

# 1. CREATE MORE SOPHISTICATED FEATURES
# Add fare per person feature
data_copy = data.copy()
if 'Fare' in train.columns and 'FamilySize' in data_copy.columns:
    data_copy['FarePerPerson'] = train['Fare'] / data_copy[:len(train)]['FamilySize']
    
# More feature engineering could be added here

# 2. SCALE FEATURES FOR SVM AND OTHER DISTANCE-BASED ALGORITHMS
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_top)
X_test_scaled = scaler.transform(X_test_top)

# 3. TRAIN MULTIPLE MODELS ON TOP FEATURES
# Random Forest (already optimized)
rf_top.fit(X_train_top, y_train)

# Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=300, max_depth=5, random_state=42)
gb.fit(X_train_top, y_train)

# Support Vector Machine
svm = SVC(C=1.0, kernel='rbf', probability=True, random_state=42)
svm.fit(X_train_scaled, y_train)

# XGBoost
xgb_model = xgb.XGBClassifier(n_estimators=200, max_depth=5, random_state=42)
xgb_model.fit(X_train_top, y_train)

# 4. CREATE ENSEMBLE USING VOTING
ensemble = VotingClassifier(estimators=[
    ('rf', rf_top),
    ('gb', gb),
    ('svm', svm),
    ('xgb', xgb_model)
], voting='soft')
ensemble.fit(X_train_top, y_train)

# 5. MAKE PREDICTIONS WITH ENSEMBLE
ensemble_predictions = ensemble.predict(X_test_top)

# 6. EVALUATE ENSEMBLE PERFORMANCE
ensemble_comparison = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "ModelPrediction": ensemble_predictions,
    "GenderSubmission": gender_submission["Survived"]
})

ensemble_accuracy = (ensemble_comparison["ModelPrediction"] == ensemble_comparison["GenderSubmission"]).mean()
print(f"Ensemble Agreement with gender_submission.csv: {ensemble_accuracy:.4f}")

# 7. CREATE SUBMISSION WITH ENSEMBLE
submission_ensemble = pd.DataFrame({
    "PassengerId": test_passenger_ids,
    "Survived": ensemble_predictions
})
submission_ensemble.to_csv("submission_ensemble.csv", index=False)
print("Ensemble submission file saved as 'submission_ensemble.csv'")

# =========================================================
# # Base models
# rf = RandomForestClassifier(n_estimators=200, max_depth=10)
# xgb_model = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.01)
# lgb_model = lgb.LGBMClassifier(n_estimators=200, max_depth=5, learning_rate=0.01)

# # Stack models with a meta-learner
# stacking = StackingClassifier(
#     estimators=[('rf', rf), ('xgb', xgb_model), ('lgb', lgb_model)],
#     final_estimator=LogisticRegression()
# )
# stacking.fit(X_train, y_train)
# # ...
# # Apply "women and children first" rule as baseline
# predictions = np.zeros(len(X_test))

# # Rule 1: Most women survived (except 3rd class)
# predictions[(test['Sex'] == 'female') & (test['Pclass'] != 3)] = 1

# # Rule 2: Rich children survived
# predictions[(test['Age'] < 12) & (test['Pclass'] < 3)] = 1

# # Use ML model only for uncertain cases
# uncertain_mask = (
#     ~((test['Sex'] == 'female') & (test['Pclass'] != 3)) & 
#     ~((test['Age'] < 12) & (test['Pclass'] < 3))
# )
# uncertain_indices = test[uncertain_mask].index
# predictions[uncertain_indices] = stacking.predict(X_test.loc[uncertain_indices])
# =========================================================
# Create a rules-based model with ML backup
# This leverages the "women and children first" pattern

# Create simple rules based on domain knowledge
X_test_rules = X_test_rf.copy()
X_test_rules['SurvivalRule'] = 0

# Rule 1: Women (except 3rd class) survived
women_not_class3 = (test['Sex'] == 'female') & (test['Pclass'] != 3)
X_test_rules.loc[women_not_class3, 'SurvivalRule'] = 1

# Rule 2: Children under 10 in 1st and 2nd class survived
children_class12 = (test['Age'] < 10) & (test['Pclass'].isin([1, 2]))
X_test_rules.loc[children_class12, 'SurvivalRule'] = 1

# Use ML model only where rules are uncertain
# Get indices where rules aren't applied
uncertain_idx = X_test_rules[X_test_rules['SurvivalRule'] == 0].index

# Train a focused model on just the cases not covered by rules
rule_based_predictions = X_test_rules['SurvivalRule'].values

# Replace predictions only for uncertain cases
rule_based_predictions[uncertain_idx] = rf_top.predict(X_test_top.loc[uncertain_idx])

# Compare with gender_submission
rule_based_comparison = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "ModelPrediction": rule_based_predictions,
    "GenderSubmission": gender_submission["Survived"]
})

rule_based_accuracy = (rule_based_comparison["ModelPrediction"] == rule_based_comparison["GenderSubmission"]).mean()
print(f"Rule-Based + ML Agreement: {rule_based_accuracy:.4f}")

rule_based_comparison.to_csv("submission_rule_based.csv", index=False)
print("Ensemble submission file saved as 'submission_rule_based.csv'")

# # ======================================= >>>> <<<< =======================================
# # Use Recursive Feature Elimination
# # Use RFE with cross-validation to find optimal feature subset
# rfecv = RFECV(
#     estimator=RandomForestClassifier(n_estimators=100, random_state=42),
#     step=1,
#     cv=5,
#     scoring='accuracy',
#     min_features_to_select=5
# )
# rfecv.fit(X_train_rf, y_train)

# print(f"Optimal number of features: {rfecv.n_features_}")
# print(f"Best features: {X_train_rf.columns[rfecv.support_]}")

# # Use only selected features
# X_train_rfe = X_train_rf.iloc[:, rfecv.support_]
# X_test_rfe = X_test_rf.iloc[:, rfecv.support_]

# # Train model on RFE-selected features
# rf_rfe = RandomForestClassifier(n_estimators=100, random_state=42)
# rf_rfe.fit(X_train_rfe, y_train)
# rfe_predictions = rf_rfe.predict(X_test_rfe)

# # Check accuracy
# rfe_comparison = pd.DataFrame({
#     "PassengerId": test["PassengerId"],
#     "ModelPrediction": rfe_predictions,
#     "GenderSubmission": gender_submission["Survived"]
# })

# rfe_accuracy = (rfe_comparison["ModelPrediction"] == rfe_comparison["GenderSubmission"]).mean()
# print(f"RFE Feature Selection Agreement: {rfe_accuracy:.4f}")