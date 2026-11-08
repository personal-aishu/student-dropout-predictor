import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, confusion_matrix, classification_report

# Configure page settings
st.set_page_config(page_title="Student Dropout Predictor", page_icon="🎓", layout="wide")

# Custom UI styling (Fixed for Dark/Light Theme Compatibility)
st.markdown("""
    <style>
    div[data-testid="stMetric"] {
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
        border-radius: 10px !important;
        padding: 15px !important;
    }
    div[data-testid="stMetric"] label {
        color: #94a3b8 !important;
        font-weight: 600 !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #38bdf8 !important;
        font-weight: 700 !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🎓 Student Dropout & Academic Success Predictor")
st.markdown("Evaluate machine learning models on their ability to predict student outcomes: **Dropout**, **Enrolled**, or **Graduate**.")

# 1. Sidebar for Data Loading & Controls
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Locate test_data.csv using script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_csv_path = os.path.join(script_dir, "test_data.csv")
    df_test = None

    # Auto-load local test_data.csv if available
    if os.path.exists(default_csv_path):
        st.success("✅ `test_data.csv` loaded automatically")
        df_test = pd.read_csv(default_csv_path)
        
        # Optional file override inside a collapsible menu
        with st.expander("📤 Upload Custom CSV (Optional)"):
            uploaded_file = st.file_uploader("Upload custom file", type=["csv"])
            if uploaded_file is not None:
                df_test = pd.read_csv(uploaded_file)
                st.info("Using custom uploaded file.")
    else:
        st.warning("⚠️ `test_data.csv` not found in project folder.")
        uploaded_file = st.file_uploader("Upload test_data.csv", type=["csv"])
        if uploaded_file is not None:
            df_test = pd.read_csv(uploaded_file)
        else:
            st.error("Run `py train_models.py` first to generate `test_data.csv`.")
            st.stop()

    # Separate features and target
    if 'Target_Encoded' in df_test.columns:
        X_test = df_test.drop('Target_Encoded', axis=1)
        y_test = df_test['Target_Encoded']
    else:
        st.error("Dataset must contain a 'Target_Encoded' column.")
        st.stop()

# 2. Load Models
@st.cache_resource
def load_models():
    model_names = ["Logistic_Regression", "Decision_Tree", "KNN", "Naive_Bayes", "Support_Vector_Machine"]
    models = {}
    for name in model_names:
        path = os.path.join(script_dir, "model", f"{name}.joblib")
        if os.path.exists(path):
            models[name] = joblib.load(path)
    return models

models = load_models()
if not models:
    st.error("No trained models found in the 'model/' directory. Please run train_models.py first.")
    st.stop()

# 3. Create Main Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Individual Model Evaluation", "🏆 Compare All Models", "🌟 Feature Insights", "🔍 Dataset Preview"])

# TAB 1: Individual Model Evaluation
with tab1:
    col1, col2 = st.columns([1, 3])
    with col1:
        selected_model_name = st.selectbox("Select Model to Evaluate", list(models.keys()))
    
    model = models[selected_model_name]
    
    # .values added here to silence sklearn warnings
    y_pred = model.predict(X_test.values)
    y_prob = model.predict_proba(X_test.values) if hasattr(model, "predict_proba") else None
    
    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    mcc = matthews_corrcoef(y_test, y_pred)
    
    # Metric Display in Columns
    st.subheader(f"Performance Metrics: {selected_model_name.replace('_', ' ')}")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", f"{acc:.4f}")
    m2.metric("Precision", f"{prec:.4f}")
    m3.metric("Recall", f"{rec:.4f}")
    m4.metric("F1-Score", f"{f1:.4f}")
    m5.metric("MCC", f"{mcc:.4f}")
    
    if y_prob is not None:
        auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
        st.metric("AUC (One-vs-Rest)", f"{auc:.4f}")

    st.markdown("---")
    
    # Visualizations layout
    viz_col1, viz_col2 = st.columns(2)
    with viz_col1:
        st.write("**Confusion Matrix**")
        fig, ax = plt.subplots(figsize=(6, 4))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Dropout', 'Enrolled', 'Graduate'], 
                    yticklabels=['Dropout', 'Enrolled', 'Graduate'])
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        st.pyplot(fig)
        
    with viz_col2:
        st.write("**Classification Report**")
        report = classification_report(y_test, y_pred, target_names=['Dropout', 'Enrolled', 'Graduate'], output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.background_gradient(cmap='viridis', subset=['f1-score']))

    # Download Predictions Button
    st.markdown("---")
    df_results = df_test.copy()
    df_results['Predicted_Target'] = y_pred
    csv_results = df_results.to_csv(index=False).encode('utf-8')
    st.download_button(
        label=f"⬇️ Download {selected_model_name.replace('_', ' ')} Predictions",
        data=csv_results,
        file_name=f"{selected_model_name}_predictions.csv",
        mime="text/csv"
    )

# TAB 2: Model Comparison Dashboard
with tab2:
    st.subheader("Leaderboard: Benchmarking All Models")
    
    comparison_data = []
    for name, clf in models.items():
        # .values added here to silence sklearn warnings
        preds = clf.predict(X_test.values)
        comparison_data.append({
            "Model": name.replace('_', ' '),
            "Accuracy": accuracy_score(y_test, preds),
            "F1 Score": f1_score(y_test, preds, average='weighted'),
            "MCC": matthews_corrcoef(y_test, preds)
        })
        
    comp_df = pd.DataFrame(comparison_data).set_index("Model")
    formatted_df = comp_df.round(4) # Clean 4-decimal formatting
    
    # Display table and chart side-by-side
    c_col1, c_col2 = st.columns([1, 2])
    with c_col1:
        st.dataframe(formatted_df.style.highlight_max(axis=0, color='#15803d'))
    with c_col2:
        st.bar_chart(formatted_df, stack=False) # Unstacked bar chart for accurate visual scale

# TAB 3: Feature Insights
with tab3:
    st.subheader("🌟 Feature Insights (Top 10 Drivers)")
    st.markdown("Visualizing which dataset features carry the most weight in predicting student dropout.")
    if "Decision_Tree" in models:
        dt_model = models["Decision_Tree"]
        if hasattr(dt_model, "feature_importances_"):
            importances = dt_model.feature_importances_
            feat_df = pd.DataFrame({"Feature": X_test.columns, "Importance": importances})
            feat_df = feat_df.sort_values(by="Importance", ascending=False).head(10)
            
            fig_feat, ax_feat = plt.subplots(figsize=(8, 4))
            # hue and legend=False added to silence seaborn future warnings
            sns.barplot(x="Importance", y="Feature", data=feat_df, hue="Feature", palette="viridis", legend=False, ax=ax_feat)
            st.pyplot(fig_feat)
    else:
        st.info("Decision Tree model required for feature importance mapping.")

# TAB 4: Dataset Preview
with tab4:
    st.subheader("Test Data Snapshot")
    st.write(f"Total test instances: {len(df_test)}")
    st.dataframe(df_test.head(20))