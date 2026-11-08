import pandas as pd
import joblib
import os
import urllib.request
import zipfile
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Create model directory if it doesn't exist
if not os.path.exists('model'):
    os.makedirs('model')

# Auto-download dataset if missing locally
if not os.path.exists('student_data.csv') and not os.path.exists('data.csv'):
    print("Downloading dataset from UCI Machine Learning Repository...")
    dataset_url = "https://archive.ics.uci.edu/static/public/697/predict+students+dropout+and+academic+success.zip"
    urllib.request.urlretrieve(dataset_url, "student_data.zip")
    with zipfile.ZipFile("student_data.zip", 'r') as zip_ref:
        zip_ref.extractall(".")

# Determine dataset filename
data_path = 'student_data.csv' if os.path.exists('student_data.csv') else 'data.csv'
print(f"Loading dataset from {data_path}...")
student_df = pd.read_csv(data_path, sep=';')

# Separate features and target label
features_X = student_df.drop('Target', axis=1)
target_y_raw = student_df['Target']

# Encode target categories (Dropout=0, Enrolled=1, Graduate=2)
target_encoder = LabelEncoder()
target_y_encoded = target_encoder.fit_transform(target_y_raw)

# Train-Test Split (80/20)
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    features_X, target_y_encoded, test_size=0.2, random_state=42
)

# Standardize features for model performance
scaler_transform = StandardScaler()
X_train_scaled = scaler_transform.fit_transform(X_train_raw)
X_test_scaled = scaler_transform.transform(X_test_raw)

# Export test dataset for Streamlit app upload
df_test_export = pd.DataFrame(X_test_scaled, columns=features_X.columns)
df_test_export['Target_Encoded'] = y_test
df_test_export.to_csv('test_data.csv', index=False)
print("Saved test_data.csv successfully.")

# Save preprocessors
joblib.dump(scaler_transform, 'model/scaler_transform.joblib')
joblib.dump(target_encoder, 'model/target_encoder.joblib')

# Define classification models
classification_models = {
    "Logistic_Regression": LogisticRegression(max_iter=3000),
    "Decision_Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive_Bayes": GaussianNB(),
    "Support_Vector_Machine": SVC(probability=True, random_state=42) 
}

# Train and export models
for algo_name, clf_model in classification_models.items():
    print(f"Currently training {algo_name}...")
    clf_model.fit(X_train_scaled, y_train)
    save_path = f"model/{algo_name}.joblib"
    joblib.dump(clf_model, save_path)
    
print("All 5 machine learning models have been trained and saved successfully!")