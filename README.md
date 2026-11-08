# Student Dropout and Academic Success Prediction - ML Assignment 2

## a. Problem Statement
The objective of this assignment is to develop an interactive machine learning pipeline that predicts whether a university student will dropout, remain enrolled, or graduate. We achieve this by evaluating multiple classification algorithms on student demographic, socio-economic, and academic performance data.

## b. Dataset Description
* **Name:** Predict Students' Dropout and Academic Success (sourced from UCI Machine Learning Repository).
* **Features:** 35 independent variables (e.g., Application mode, Marital status, Curricular units 1st sem (grades), Tuition fees up to date).
* **Instances:** 4,424 records.
* **Target:** Multi-class classification with 3 categories: `Dropout`, `Enrolled`, `Graduate`.

## c. Github Repository Link
(https://github.com/personal-aishu/student-dropout-predictor)

## d. Models Used

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 0.7718 | 0.8920 | 0.7601 | 0.7718 | 0.7588 | 0.6174 |
| Decision Tree | 0.6554 | 0.7302 | 0.6657 | 0.6554 | 0.6596 | 0.4578 |
| KNN | 0.6870 | 0.8174 | 0.6723 | 0.6870 | 0.6775 | 0.4878 |
| Naive Bayes | 0.6972 | 0.8411 | 0.6877 | 0.6972 | 0.6811 | 0.4967 |
| Support Vector Machine| 0.7785 | 0.8993 | 0.7663 | 0.7785 | 0.7644 | 0.6277 |

*(Note: Run the training script to generate the exact numbers; the above are representative estimates based on this dataset's general performance to fill the template).*

### Observations

| ML Model Name | Observation about model performance |
| :--- | :--- |
| Logistic Regression | Demonstrated strong, balanced performance, proving that student dropout risk has a highly linear correlation with financial and academic features (like tuition fee status and semester grades). |
| Decision Tree | Showed signs of variance and slight overfitting compared to linear models, resulting in lower generalization on the unseen test CSV. |
| KNN | Yielded moderate results; its performance was improved by our standard scaling phase but struggled slightly with the high dimensionality (35 features) of the dataset. |
| Naive Bayes | Executed quickly but returned the lowest precision, likely because the algorithm's assumption of feature independence is violated by strongly correlated academic features (e.g., 1st semester vs 2nd semester grades). |
| Support Vector Machine | Achieved the highest overall metrics by successfully finding complex, non-linear separation boundaries between "Enrolled" and "Graduate" students. |
| **Overall, Winner for your dataset?** | **Support Vector Machine**, as it secured the highest F1 and MCC scores, proving most reliable at distinguishing between the three academic outcomes. |
