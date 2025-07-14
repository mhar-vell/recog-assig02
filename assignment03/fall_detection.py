import os
import numpy as np
import pandas as pd
import scipy.io
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.metrics import classification_report
import shap
import plotly.express as px

# 1. Feature extraction from .mat files
def extract_features_from_file(filepath):
    mat = scipy.io.loadmat(filepath)
    data = mat['newData']
    imu_data = data[:, 1:10]  # Skip time and ignore final column
    means = np.mean(imu_data, axis=0)
    max_vals = np.max(imu_data, axis=0)
    std_devs = np.std(imu_data, axis=0)
    return np.concatenate([means, max_vals, std_devs])

# 2. Build dataset from folder
def process_dataset(root_folder):
    feature_list = []
    label_list = []
    for label_folder, label in [('falls', 2), ('nonfalls', 1)]:
        folder_path = os.path.join(root_folder, label_folder)
        for file in os.listdir(folder_path):
            if file.endswith('.mat'):
                path = os.path.join(folder_path, file)
                feat = extract_features_from_file(path)
                feature_list.append(feat)
                label_list.append(label)
    return np.array(feature_list), np.array(label_list)

# 3. Load training/testing data
X_train, y_train = process_dataset("path_to_training_folder")
X_test, y_test = process_dataset("path_to_testing_folder")

# 4. Train models
rf = RandomForestClassifier().fit(X_train, y_train)
gb = GradientBoostingClassifier().fit(X_train, y_train)
svm = SVC(probability=True).fit(X_train, y_train)
xgb_model = xgb.XGBClassifier().fit(X_train, y_train)

# 5. Generate SHAP values
explainer_rf = shap.Explainer(rf, X_train)
explainer_gb = shap.Explainer(gb, X_train)
explainer_xgb = shap.Explainer(xgb_model, X_train)
explainer_svm = shap.KernelExplainer(svm.predict_proba, X_train)

shap_rf = explainer_rf(X_test)
shap_gb = explainer_gb(X_test)
shap_xgb = explainer_xgb(X_test)
shap_svm = explainer_svm.shap_values(X_test)[1]  # Class 1 SHAP values

# 6. Feature names for plotting
feature_names = [f"{stat}_{sensor}_{axis}" for stat in ['mean','max','std']
                 for sensor in ['acc','gyro','mag'] for axis in ['x','y','z']]

# 7. Helper: Get top SHAP values
def top_shap(shap_vals, model_name, feature_names):
    mean_abs = np.abs(shap_vals).mean(axis=0)
    return pd.DataFrame({
        'Feature': feature_names,
        model_name: mean_abs
    }).sort_values(by=model_name, ascending=False)

# 8. Build comparison table
df_rf = top_shap(shap_rf.values, 'Random Forest', feature_names)
df_gb = top_shap(shap_gb.values, 'Gradient Boosting', feature_names)
df_xgb = top_shap(shap_xgb.values, 'XGBoost', feature_names)
df_svm = top_shap(shap_svm, 'SVM', feature_names)

df_merge = df_rf.merge(df_gb, on='Feature', how='outer') \
                .merge(df_xgb, on='Feature', how='outer') \
                .merge(df_svm, on='Feature', how='outer')

# 9. Melt for Plotly visualization
df_melt = df_merge.melt(id_vars='Feature', var_name='Model', value_name='SHAP Value')

# 10. Plotly magic 💫
fig = px.bar(
    df_melt, x='SHAP Value', y='Feature', color='Model',
    orientation='h', barmode='group',
    title='🎯 SHAP Feature Importance Battle Royale',
    labels={'SHAP Value': 'Mean |SHAP| Value'}
)

fig.update_layout(
    legend_title_text='Judge Model',
    xaxis_title='Impact on Prediction',
    yaxis_title='Sensor Feature',
    template='plotly_dark'
)

fig.show()

# 11. Optional: Evaluate performance
y_pred = rf.predict(X_test)
print("🔍 Random Forest Classification Report:\n")
print(classification_report(y_test, y_pred))
