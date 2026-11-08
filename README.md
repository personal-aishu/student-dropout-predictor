# Student Dropout & Academic Success Prediction 🎓

## 1. Problem Statement
Higher education institutions face significant challenges with student attrition. Early identification of students at risk of dropping out allows academic advisors and administration to provide targeted support, interventions, and counseling. 

This project builds and benchmarks five distinct supervised machine learning classification algorithms (**Logistic Regression**, **Decision Tree**, **K-Nearest Neighbors**, **Gaussian Naive Bayes**, and **Support Vector Machine**) to predict student academic outcomes (**Dropout**, **Enrolled**, or **Graduate**) using demographic, socio-economic, and academic data collected at enrollment and during early semesters.

<<<<<<< HEAD
---
=======
## c. Github Repository Link
(https://github.com/personal-aishu/student-dropout-predictor)
>>>>>>> cf97f845748b78e4830a359ea4c65da6cdb54cb4

## 2. Dataset Overview
* **Dataset Name:** Predict Students' Dropout and Academic Success
* **Source:** UCI Machine Learning Repository
* **Total Instances:** 4,424 records
* **Total Features:** 35 independent variables
* **Target Variable:** `Target` (`Dropout`, `Enrolled`, `Graduate`)

### Key Feature Categories:
- **Academic Performance:** 1st/2nd semester credited units, evaluated units, approved units, grade averages, and evaluations.
- **Socio-Economic Indicators:** Mother's and Father's qualification/occupation, tuition fee payment status, scholarship holder status, and debtor status.
- **Demographics & Background:** Age at enrollment, gender, marital status, international status, application mode, and course of study.

---

## 3. Machine Learning Pipeline & Methodology
1. **Data Preprocessing & Encoding:** Handled missing data, converted categorical values, and applied Label Encoding to target classes (`Dropout = 0`, `Enrolled = 1`, `Graduate = 2`).
2. **Feature Scaling:** Applied `StandardScaler` across continuous numerical variables to optimize distance-based and gradient-based algorithms (KNN, Logistic Regression, SVM).
3. **Train-Test Split:** Created a stratified 80:20 train-test split to preserve class proportions.
4. **Model Serialization:** Trained models are serialized into `.joblib` artifacts in the `model/` directory for fast inference without re-training during application startup.

---

## 4. Model Benchmarking & Performance

Below is the comparative evaluation across six standard classification metrics evaluated on the test set:

| ML Model Name | Accuracy | AUC (OvR) | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.7525 | 0.8845 | 0.7410 | 0.7525 | 0.7363 | 0.5932 |
| **Decision Tree** | 0.6757 | 0.7321 | 0.6812 | 0.6757 | 0.6780 | 0.4798 |
| **KNN** | 0.7017 | 0.7924 | 0.6927 | 0.7017 | 0.6912 | 0.5107 |
| **Naive Bayes** | 0.6927 | 0.8351 | 0.6877 | 0.6927 | 0.6798 | 0.4943 |
| **Support Vector Machine (SVM)** | **0.7582** | **0.8912** | **0.7490** | **0.7582** | **0.7441** | **0.6063** |

---

## 5. Model Observations & Analytical Insights

| Model | Performance Observation |
| :--- | :--- |
| **Logistic Regression** | Demonstrated strong, balanced performance, proving that student dropout risk has a highly linear correlation with financial and academic features (like tuition fee status and semester grades). |
| **Decision Tree** | Showed signs of variance and slight overfitting compared to linear models, resulting in lower generalization on unseen test data. |
| **KNN** | Yielded moderate results; performance improved after standard scaling but struggled slightly with the high dimensionality (35 features) of the dataset. |
| **Naive Bayes** | Executed quickly but returned lower precision, likely because the algorithm's assumption of feature independence is violated by strongly correlated academic features (e.g., 1st semester vs 2nd semester grades). |
| **Support Vector Machine** | Achieved the highest overall metrics by successfully finding complex, non-linear separation boundaries between "Enrolled" and "Graduate" students. |
| **Overall Winner** | **Support Vector Machine (SVM)** achieved the top F1-Score (**0.7441**) and Matthews Correlation Coefficient (**0.6063**), making it the most reliable classifier for this dataset. |

---

## 6. Interactive Streamlit Dashboard (`app.py`)
The web application features an interactive 4-tab interface designed for comprehensive model exploration:

* **📊 Individual Model Evaluation:** Select any model to review accuracy cards, weighted precision/recall/F1 metrics, confusion matrices (Seaborn heatmaps), and full classification reports. Includes a **Download Predictions** CSV feature.
* **🏆 Compare All Models:** Displays a live benchmarking leaderboard comparing Accuracy, F1 Score, and MCC side-by-side using an unstacked Streamlit bar chart.
* **🌟 Feature Insights:** Visualizes the Top 10 predictive drivers (such as approved curricular units and tuition fee payment status) derived from feature importance coefficients.
* **🔍 Dataset Preview & Auto-Load:** Automatically detects and loads local `test_data.csv` artifacts without requiring manual drag-and-drop file uploads, while maintaining an optional custom CSV file uploader menu.

---

## 7. Project Structure
```text
ML_Assignment_2/
├── model/
│   ├── Decision_Tree.joblib
│   ├── KNN.joblib
│   ├── Logistic_Regression.joblib
│   ├── Naive_Bayes.joblib
│   ├── Support_Vector_Machine.joblib
│   ├── scaler_transform.joblib
│   └── target_encoder.joblib
├── app.py
├── train_models.py
├── test_data.csv
├── requirements.txt
└── README.md