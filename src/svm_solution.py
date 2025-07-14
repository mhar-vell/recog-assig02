import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
def load_data(train_path, test_path):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    return train_data, test_data

# Preprocess the data
def preprocess_data(data):
    # Fill missing values
    data['Age'].fillna(data['Age'].median(), inplace=True)
    data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
    
    # Encode categorical variables
    data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
    data = pd.get_dummies(data, columns=['Embarked'], drop_first=True)
    
    # Drop unnecessary columns
    data.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1, inplace=True, errors='ignore')
    
    return data

# Train the SVM model
def train_svm(X_train, y_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    svm_model = SVC(kernel='linear', random_state=42)
    svm_model.fit(X_train_scaled, y_train)
    
    return svm_model, scaler

# Evaluate the model
def evaluate_model(model, scaler, X_test, y_test):
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

# Main function to run the SVM solution
def main():
    train_data, test_data = load_data('data/train.csv', 'data/test.csv')
    
    # Separate features and target variable
    X = preprocess_data(train_data.drop('Survived', axis=1))
    y = train_data['Survived']
    X_test = preprocess_data(test_data.copy())
    
    # Align columns
    X, X_test = X.align(X_test, join='left', axis=1, fill_value=0)
    
    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the SVM model
    svm_model, scaler = train_svm(X_train, y_train)
    
    # Evaluate the model
    evaluate_model(svm_model, scaler, X_val, y_val)

if __name__ == "__main__":
    main()