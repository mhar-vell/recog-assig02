import scipy.io
import numpy as np
import os

def extract_features_from_file(filepath):
    mat = scipy.io.loadmat(filepath)
    data = mat['newData']  # shape should be [500, 11]

    # Get the 9 sensor signals (columns 1 to 9)
    imu_data = data[:, 1:10]

    # Feature extraction
    means = np.mean(imu_data, axis=0)
    max_values = np.max(imu_data, axis=0)
    std_devs = np.std(imu_data, axis=0)

    # Combine all features into one vector
    feature_vector = np.concatenate([means, max_values, std_devs])
    return feature_vector

def process_dataset(root_folder):
    feature_list = []
    label_list = []

    for label_folder, label in [('falls', 2), ('nonfalls', 1)]:
        folder_path = os.path.join(root_folder, label_folder)
        for filename in os.listdir(folder_path):
            if filename.endswith('.mat'):
                filepath = os.path.join(folder_path, filename)
                features = extract_features_from_file(filepath)
                feature_list.append(features)
                label_list.append(label)

    X = np.array(feature_list)
    y = np.array(label_list)
    return X, y


### classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load your training data
X_train, y_train = process_dataset('path_to_training_folder')
X_test, y_test = process_dataset('path_to_testing_folder')

# Grid search for best hyperparameters
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best model
best_rf = grid_search.best_estimator_

# Evaluate on test set
y_pred = best_rf.predict(X_test)

# Print results
print("Best Parameters:", grid_search.best_params_)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# Save the model
import joblib
model_filename = 'best_random_forest_model.pkl'
joblib.dump(best_rf, model_filename)    


# explore more feature important
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load your training data
X_train, y_train = process_dataset('path_to_training_folder')
X_test, y_test = process_dataset('path_to_testing_folder')

# Grid search for best hyperparameters
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best model
best_rf = grid_search.best_estimator_

# Evaluate on test set
y_pred = best_rf.predict(X_test)

# Print results
print("Best Parameters:", grid_search.best_params_)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load your training data
X_train, y_train = process_dataset('path_to_training_folder')
X_test, y_test = process_dataset('path_to_testing_folder')

# Grid search for best hyperparameters
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best model
best_rf = grid_search.best_estimator_

# Evaluate on test set
y_pred = best_rf.predict(X_test)

# Print results
print("Best Parameters:", grid_search.best_params_)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# Save the submission file
submission = pd.DataFrame({
    "Id": np.arange(len(y_pred)),
    "Predicted": y_pred     

})
submission.to_csv('submission.csv', index=False)
print(f"Submission file saved as 'submission.csv'") 