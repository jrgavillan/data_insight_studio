import streamlit as st
import requests
import base64
import json
import os
from datetime import datetime
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import ttest_ind, ttest_rel, f_oneway, chi2_contingency, linregress, norm
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

st.set_page_config(page_title="Data Insight Studio", layout="wide", initial_sidebar_state="expanded")

CONFIG_FILE = "api_config.txt"
PRIVACY_POLICY_URL = "https://jrgavillan.github.io/data_insight_studio/privacy_policy.md"
TERMS_URL = "https://jrgavillan.github.io/data_insight_studio/privacy_policy.md"

API_COSTS = {
    "per_1m_input": 0.003,
    "per_1m_output": 0.015,
    "avg_input_tokens": 500,
    "avg_output_tokens": 300
}

PRICING = {
    "per_term": 14.99,
    "term_days": 90
}

# All available analysis categories
ANALYSIS_CATEGORIES = {
    "📊 DESCRIPTIVE STATISTICS": "Descriptive Statistics",
    "📈 HYPOTHESIS TESTING": "Hypothesis Testing",
    "📉 REGRESSION ANALYSIS": "Regression",
    "📌 ANOVA": "ANOVA",
    "📊 CONFIDENCE INTERVALS": "Confidence Intervals",
    "🔔 PROBABILITY DISTRIBUTIONS": "Probability",
    "🧪 T-TESTS": "T-Tests",
    "🎯 CHI-SQUARE TEST": "Chi-Square",
    "🤖 MACHINE LEARNING": "Machine Learning",
    "🔮 PREDICTIVE MODELING": "Predictive Modeling",
    "📊 CLUSTERING ANALYSIS": "Clustering",
    "📈 TIME SERIES": "Time Series",
    "🎨 DATA VISUALIZATION": "Data Visualization",
    "📋 CORRELATION ANALYSIS": "Correlation",
}

# ============================================================================
# PERSISTENT API KEY STORAGE
# ============================================================================

def save_api_key(key):
    try:
        with open(CONFIG_FILE, "w") as f:
            f.write(key)
        return True
    except:
        return False

def load_api_key():
    env_key = os.environ.get("ANTHROPIC_API_KEY")
    if env_key:
        return env_key
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r") as f:
                return f.read().strip()
    except:
        pass
    return None

def get_api_key():
    if st.session_state.api_key:
        return st.session_state.api_key
    file_key = load_api_key()
    if file_key:
        st.session_state.api_key = file_key
        return file_key
    return None

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def image_to_base64(image_file):
    return base64.b64encode(image_file.read()).decode()

def read_excel_csv(file):
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
            return df
        elif file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file)
            return df
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return None

def calculate_descriptive_stats(df, column):
    stats_dict = {
        "Mean": df[column].mean(),
        "Median": df[column].median(),
        "Std Dev": df[column].std(),
        "Min": df[column].min(),
        "Max": df[column].max(),
        "Q1": df[column].quantile(0.25),
        "Q3": df[column].quantile(0.75),
        "IQR": df[column].quantile(0.75) - df[column].quantile(0.25),
        "Skewness": df[column].skew(),
        "Kurtosis": df[column].kurtosis(),
        "N": len(df)
    }
    return stats_dict

def create_visualizations(df, column):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0, 0].hist(df[column].dropna(), bins=30, edgecolor='black', alpha=0.7)
    axes[0, 0].set_title(f'Histogram of {column}')
    axes[0, 0].set_xlabel('Value')
    axes[0, 0].set_ylabel('Frequency')
    
    axes[0, 1].boxplot(df[column].dropna())
    axes[0, 1].set_title(f'Box Plot of {column}')
    axes[0, 1].set_ylabel('Value')
    
    stats.probplot(df[column].dropna(), dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot')
    
    df[column].plot(kind='density', ax=axes[1, 1])
    axes[1, 1].set_title(f'Density Plot of {column}')
    
    plt.tight_layout()
    return fig

def analyze_hypothesis_testing(df):
    st.write("### Hypothesis Testing Analysis")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 1:
        st.warning("Need at least 1 numeric column")
        return None
    
    col1, col2 = st.columns(2)
    with col1:
        test_type = st.selectbox("Test Type", ["One-sample t-test", "Two-sample t-test", "Chi-square"])
    
    results = {}
    
    if test_type == "One-sample t-test":
        col = st.selectbox("Select column", numeric_cols, key="hyp_col")
        null_mean = st.number_input("Null hypothesis mean", value=0.0)
        t_stat, p_value = stats.ttest_1samp(df[col].dropna(), null_mean)
        results = {
            "Test": "One-sample t-test",
            "Null Mean": null_mean,
            "t-statistic": t_stat,
            "p-value": p_value,
            "Significant": "Yes" if p_value < 0.05 else "No"
        }
    elif test_type == "Two-sample t-test":
        if len(numeric_cols) >= 2:
            col1_select = st.selectbox("Column 1", numeric_cols, key="col1")
            col2_select = st.selectbox("Column 2", numeric_cols, key="col2", index=1)
            t_stat, p_value = ttest_ind(df[col1_select].dropna(), df[col2_select].dropna())
            results = {
                "Test": "Two-sample t-test",
                "Column 1": col1_select,
                "Column 2": col2_select,
                "t-statistic": t_stat,
                "p-value": p_value,
                "Significant": "Yes" if p_value < 0.05 else "No"
            }
    
    if results:
        st.write("**Results:**")
        results_df = pd.DataFrame(list(results.items()), columns=['Metric', 'Value'])
        st.dataframe(results_df, use_container_width=True)
        return str(results)
    return None

def analyze_regression(df):
    st.write("### Regression Analysis")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.warning("Need at least 2 numeric columns")
        return None
    
    x_col = st.selectbox("Independent Variable (X)", numeric_cols, key="x_col")
    y_col = st.selectbox("Dependent Variable (Y)", numeric_cols, key="y_col", index=1)
    
    data_clean = df[[x_col, y_col]].dropna()
    X = data_clean[x_col].values.reshape(-1, 1)
    y = data_clean[y_col].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    r_squared = model.score(X, y)
    slope = model.coef_[0]
    intercept = model.intercept_
    
    y_pred = model.predict(X)
    residuals = y - y_pred
    rss = np.sum(residuals**2)
    mse = rss / (len(y) - 2)
    se_slope = np.sqrt(mse / np.sum((X - X.mean())**2))
    t_stat = slope / se_slope
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(y) - 2))
    
    results = {
        "Equation": f"Y = {intercept:.4f} + {slope:.4f}*X",
        "R-squared": r_squared,
        "Slope": slope,
        "Intercept": intercept,
        "p-value (slope)": p_value,
        "Significant": "Yes" if p_value < 0.05 else "No"
    }
    
    st.write("**Regression Results:**")
    results_df = pd.DataFrame(list(results.items()), columns=['Metric', 'Value'])
    st.dataframe(results_df, use_container_width=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X, y, alpha=0.6, label='Data')
    X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_line = model.predict(X_line)
    ax.plot(X_line, y_line, 'r-', label=f'Fit: Y = {intercept:.2f} + {slope:.2f}*X')
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title('Linear Regression')
    ax.legend()
    st.pyplot(fig)
    
    return str(results)

def analyze_machine_learning(df):
    st.write("### Machine Learning Analysis")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.warning("Need at least 2 numeric columns")
        return None
    
    col1, col2 = st.columns(2)
    with col1:
        ml_type = st.selectbox("ML Algorithm", 
            ["Random Forest Regression", "Random Forest Classification", 
             "Gradient Boosting", "Support Vector Machine"])
    
    with col2:
        target_col = st.selectbox("Target Variable (Y)", numeric_cols, key="ml_target")
    
    # Feature selection
    feature_cols = [col for col in numeric_cols if col != target_col]
    if not feature_cols:
        st.warning("Need at least 2 columns")
        return None
    
    X = df[feature_cols].fillna(df[feature_cols].mean())
    y = df[target_col].fillna(df[target_col].mean())
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if ml_type == "Random Forest Regression":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results = {
            "Algorithm": "Random Forest Regression",
            "R² Score": r2,
            "MSE": mse,
            "RMSE": np.sqrt(mse),
            "Training Samples": len(X_train),
            "Test Samples": len(X_test)
        }
        
        st.write("**Model Performance:**")
        results_df = pd.DataFrame(list(results.items()), columns=['Metric', 'Value'])
        st.dataframe(results_df, use_container_width=True)
        
        # Feature importance
        importances = model.feature_importances_
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(feature_cols, importances)
        ax.set_xlabel('Importance')
        ax.set_title('Feature Importance - Random Forest')
        st.pyplot(fig)
        
        return str(results)
    
    return None

def analyze_predictive_modeling(df):
    st.write("### Predictive Modeling")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.warning("Need at least 2 numeric columns")
        return None
    
    target_col = st.selectbox("Target Variable", numeric_cols, key="pred_target")
    feature_cols = [col for col in numeric_cols if col != target_col]
    
    X = df[feature_cols].fillna(df[feature_cols].mean())
    y = df[target_col].fillna(df[target_col].mean())
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    results = {
        "Train R²": train_r2,
        "Test R²": test_r2,
        "Train RMSE": train_rmse,
        "Test RMSE": test_rmse,
        "Overfitting": "Yes" if train_r2 - test_r2 > 0.1 else "No"
    }
    
    st.write("**Predictive Model Performance:**")
    results_df = pd.DataFrame(list(results.items()), columns=['Metric', 'Value'])
    st.dataframe(results_df, use_container_width=True)
    
    # Prediction vs Actual
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.scatter(y_train, y_pred_train, alpha=0.5, label='Training')
    ax1.scatter(y_test, y_pred_test, alpha=0.5, label='Testing')
    ax1.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    ax1.set_xlabel('Actual Values')
    ax1.set_ylabel('Predicted Values')
    ax1.set_title('Predictions vs Actual')
    ax1.legend()
    
    residuals = y_test - y_pred_test
    ax2.scatter(y_pred_test, residuals, alpha=0.5)
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.set_xlabel('Predicted Values')
    ax2.set_ylabel('Residuals')
    ax2.set_title('Residual Plot')
    
    st.pyplot(fig)
    return str(results)

def analyze_clustering(df):
    st.write("### Clustering Analysis (K-Means)")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.warning("Need at least 2 numeric columns")
        return None
    
    n_clusters = st.slider("Number of Clusters", 2, 10, 3)
    
    X = df[numeric_cols].fillna(df[numeric_cols].mean())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    inertia = kmeans.inertia_
    
    results = {
        "Number of Clusters": n_clusters,
        "Inertia": inertia,
        "Samples": len(df)
    }
    
    st.write("**Clustering Results:**")
    results_df = pd.DataFrame(list(results.items()), columns=['Metric', 'Value'])
    st.dataframe(results_df, use_container_width=True)
    
    # Visualization
    if len(numeric_cols) >= 2:
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(X[numeric_cols[0]], X[numeric_cols[1]], c=clusters, cmap='viridis', s=50)
        ax.scatter(X.iloc[kmeans.cluster_centers_[:, 0]], 
                   X.iloc[kmeans.cluster_centers_[:, 1]], 
                   c='red', s=200, alpha=0.8, marker='X', label='Centroids')
        ax.set_xlabel(numeric_cols[0])
        ax.set_ylabel(numeric_cols[1])
        ax.set_title(f'K-Means Clustering (k={n_clusters})')
        plt.colorbar(scatter, ax=ax, label='Cluster')
        st.pyplot(fig)
    
    return str(results)

def analyze_anova(df):
    st.write("### ANOVA Analysis")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if len(numeric_cols) < 1 or len(cat_cols) < 1:
        st.warning("Need 1+ numeric and 1+ categorical columns")
        return None
    
    numeric_col = st.selectbox("Numeric Variable", numeric_cols, key="anova_num")
    categorical_col = st.selectbox("Categorical Variable", cat_cols, key="anova_cat")
    
    groups = [group[numeric_col].dropna().values for name, group in df.groupby(categorical_col)]
    f_stat, p_value = f_oneway(*groups)
    
    results = {
        "F-statistic": f_stat,
        "p-value": p_value,
        "Number of Groups": len(groups),
        "Significant": "Yes" if p_value < 0.05 else "No"
    }
    
    st.write("**ANOVA Results:**")
    results_df = pd.DataFrame(list(results.items()), columns=['Metric', 'Value'])
    st.dataframe(results_df, use_container_width=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    df.boxplot(column=numeric_col, by=categorical_col, ax=ax)
    ax.set_title(f'{numeric_col} by {categorical_col}')
    st.pyplot(fig)
    
    return str(results)

def analyze_correlation(df):
    st.write("### Correlation Analysis")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.warning("Need at least 2 numeric columns")
        return None
    
    corr_matrix = df[numeric_cols].corr()
    
    st.write("**Correlation Matrix:**")
    st.dataframe(corr_matrix, use_container_width=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
    ax.set_title('Correlation Heatmap')
    st.pyplot(fig)
    
    return str(corr_matrix)

def calculate_api_cost(input_tokens, output_tokens):
    input_cost = (input_tokens / 1_000_000) * API_COSTS["per_1m_input"]
    output_cost = (output_tokens / 1_000_000) * API_COSTS["per_1m_output"]
    return input_cost + output_cost

def estimate_problem_cost():
    return calculate_api_cost(API_COSTS["avg_input_tokens"], API_COSTS["avg_output_tokens"])

def solve_problem_with_ai(problem_text, category, api_key, image_data=None, learning_mode=False):
    try:
        if not api_key:
            st.error("API key is missing!")
            return None
        
        if not api_key.startswith("sk-ant-"):
            st.error("Invalid API key format")
            return None
        
        if not image_data and (not problem_text or problem_text.strip() == ""):
            st.error("Please enter a problem, upload an image, or upload a file")
            return None
        
        content = []
        
        if image_data:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": image_data
                }
            })
        
        system_prompt = f"""You are an expert statistics and data science tutor. Your goal is to help students UNDERSTAND concepts.

IMPORTANT RULES:
1. ALWAYS show step-by-step work
2. EXPLAIN the WHY behind each step
3. Use LaTeX for formulas: $$formula$$
4. NEVER give just an answer - always include reasoning

Category: {category}
Learning Mode: {'Yes - Provide hints first' if learning_mode else 'Standard - Full explanation'}"""
        
        text_prompt = f"""Analyze and explain this statistics/data problem step-by-step.

PROBLEM: {problem_text}

RULES:
- Show ALL work
- Use LaTeX for formulas
- Explain each step
- State assumptions
- Provide interpretation"""
        
        content.append({"type": "text", "text": text_prompt})

        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
            json={
                "model": "claude-opus-4-1-20250805",
                "max_tokens": 2000,
                "system": system_prompt,
                "messages": [{"role": "user", "content": content}]
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            solution_text = result['content'][0]['text']
            
            input_tokens = API_COSTS["avg_input_tokens"]
            output_tokens = API_COSTS["avg_output_tokens"]
            cost = calculate_api_cost(input_tokens, output_tokens)
            
            st.session_state.api_calls += 1
            st.session_state.api_costs += cost
            st.session_state.usage_log.append({
                "timestamp": datetime.now(),
                "problem": problem_text[:50] if problem_text else "Image",
                "category": category,
                "cost": cost,
            })
            
            return solution_text
        else:
            st.error(f"API Error {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

# ============================================================================
# SESSION STATE
# ============================================================================

if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "user_name" not in st.session_state:
    st.session_state.user_name = None
if "api_key" not in st.session_state:
    st.session_state.api_key = None
if "api_calls" not in st.session_state:
    st.session_state.api_calls = 0
if "api_costs" not in st.session_state:
    st.session_state.api_costs = 0.0
if "usage_log" not in st.session_state:
    st.session_state.usage_log = []
if "current_page" not in st.session_state:
    st.session_state.current_page = "home"
if "terms_accepted" not in st.session_state:
    st.session_state.terms_accepted = False

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    try:
        st.image("logo_1.png", width=250)
    except:
        st.title("Data Insight Studio")
    st.divider()
    
    if not st.session_state.user_id:
        login_type = st.radio("Login as:", ["Student", "Admin"], key="login_type")
        
        if login_type == "Student":
            st.write("### Student Access")
            st.write("**Demo:**")
            st.write("Email: student@example.com")
            st.write("Pass: password")
            st.write("")
            
            student_email = st.text_input("Email:", key="student_email")
            student_pass = st.text_input("Password:", type="password", key="student_pass")
            terms_check = st.checkbox("I agree to Terms & Privacy", key="terms_agree")
            
            if st.button("Sign In", key="student_signin"):
                if not terms_check:
                    st.error("Accept terms to continue")
                elif student_email == "student@example.com" and student_pass == "password":
                    st.session_state.user_id = f"student_{student_email}"
                    st.session_state.user_name = "Student"
                    st.session_state.terms_accepted = True
                    st.rerun()
                else:
                    st.error("Invalid credentials")
        else:
            st.write("### Admin Access")
            admin_pass = st.text_input("Password:", type="password", key="admin_pass")
            
            if st.button("Sign In", key="admin_signin"):
                if admin_pass == "admin123":
                    st.session_state.user_id = "admin"
                    st.session_state.user_name = "Admin"
                    st.rerun()
                else:
                    st.error("Invalid password")
        
        st.divider()
        st.write("### Legal")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("[Privacy](https://jrgavillan.github.io/data_insight_studio/privacy_policy.md)")
        with col2:
            st.markdown("[Terms](https://jrgavillan.github.io/data_insight_studio/privacy_policy.md)")
    else:
        st.write(f"### {st.session_state.user_name}! 👋")
        st.divider()
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Home", use_container_width=True):
                st.session_state.current_page = "home"
                st.rerun()
            if st.button("Resources", use_container_width=True):
                st.session_state.current_page = "resources"
                st.rerun()
        with col2:
            if st.button("Homework Help", use_container_width=True):
                st.session_state.current_page = "homework"
                st.rerun()
            if st.button("Analytics", use_container_width=True):
                st.session_state.current_page = "analytics"
                st.rerun()
        
        st.divider()
        if st.button("Sign Out", use_container_width=True):
            st.session_state.user_id = None
            st.session_state.user_name = None
            st.rerun()

# ============================================================================
# MAIN CONTENT
# ============================================================================

if not st.session_state.user_id:
    st.error("Please sign in to continue")
else:
    current_page = st.session_state.current_page
    
    if current_page == "home":
        st.title("📊 Data Insight Studio")
        st.subheader("Professional Statistics & Machine Learning Homework Helper")
        st.divider()
        
        # Hero Section
        col1, col2 = st.columns([2, 1])
        with col1:
            st.write("""
            ### Welcome to Your Statistics & ML Companion! 🎓
            
            Master statistics, data analysis, and machine learning with AI-powered guidance.
            Upload your data, choose your analysis, and learn step-by-step!
            
            **Pricing:** $14.99 per 90-day term  
            **Support:** Questions? Contact us anytime
            """)
        
        with col2:
            st.info("⚖️ **Academic Integrity**\n\nDo NOT use on proctored exams!")
        
        st.divider()
        
        # All Analysis Options as Beautiful Cards
        st.write("## 🚀 Available Analysis & Tools")
        st.write("Click **Homework Help** to get started with any of these:")
        
        # Display all options in a grid
        cols = st.columns(2)
        analysis_list = list(ANALYSIS_CATEGORIES.items())
        
        for idx, (display_name, category) in enumerate(analysis_list):
            with cols[idx % 2]:
                st.write(f"### {display_name}")
                if category == "Descriptive Statistics":
                    st.write("📈 Mean, median, std dev, visualizations")
                elif category == "Hypothesis Testing":
                    st.write("🧪 t-tests, chi-square, significance testing")
                elif category == "Regression":
                    st.write("📉 Linear/polynomial regression, R², predictions")
                elif category == "ANOVA":
                    st.write("📌 One-way ANOVA, group comparisons")
                elif category == "Confidence Intervals":
                    st.write("📊 CI calculation, margin of error")
                elif category == "Probability":
                    st.write("🔔 Distributions, normality testing")
                elif category == "T-Tests":
                    st.write("🧪 One/two-sample, paired t-tests")
                elif category == "Chi-Square":
                    st.write("🎯 Categorical association, independence")
                elif category == "Machine Learning":
                    st.write("🤖 Random Forest, Gradient Boosting, feature importance")
                elif category == "Predictive Modeling":
                    st.write("🔮 Train/test splits, prediction accuracy, overfitting")
                elif category == "Clustering":
                    st.write("📊 K-Means, cluster analysis, inertia")
                elif category == "Time Series":
                    st.write("📈 Trends, seasonality, forecasting")
                elif category == "Data Visualization":
                    st.write("🎨 Plots, heatmaps, dashboards")
                elif category == "Correlation":
                    st.write("📋 Pearson/Spearman, correlation matrix")
        
        st.divider()
        
        # Key Features
        st.write("## ✨ Key Features")
        
        feature_cols = st.columns(3)
        with feature_cols[0]:
            st.write("""
            ### 🧠 Smart Analysis
            - Auto-detect appropriate tests
            - Instant calculations
            - Professional visualizations
            """)
        
        with feature_cols[1]:
            st.write("""
            ### 📚 Learn with AI
            - Step-by-step explanations
            - Learning mode (hints first)
            - Plain language interpretation
            """)
        
        with feature_cols[2]:
            st.write("""
            ### 🎯 Complete Toolkit
            - 14+ analysis types
            - ML algorithms
            - Data visualization
            """)
        
        st.divider()
        
        # Call to Action
        st.write("## 🎯 Get Started Now!")
        st.write("Select **Homework Help** in the sidebar to:")
        st.write("1. Upload your CSV/Excel file or image")
        st.write("2. Choose your analysis type")
        st.write("3. Get instant statistical results")
        st.write("4. Ask AI to explain the results")
        st.write("5. Learn and master statistics!")
        
        st.success("Ready? Click **Homework Help** in the sidebar! 🚀")
        
        st.divider()
        st.subheader("Privacy & Legal")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("[Privacy Policy](https://jrgavillan.github.io/data_insight_studio/privacy_policy.md)")
        with col2:
            st.markdown("[Terms of Service](https://jrgavillan.github.io/data_insight_studio/privacy_policy.md)")
        with col3:
            st.markdown("[Contact Us](mailto:privacy@datainsightstudio.com)")
        
        st.info("🔒 We process and delete uploads. No storage. No training use.")
    
    elif current_page == "homework":
        st.header("📚 Homework Help")
        st.write("Upload data or image, select analysis type, get instant results!")
        st.divider()
        
        current_api_key = get_api_key()
        
        if not current_api_key:
            st.warning("System not configured. Contact support.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                learning_mode = st.checkbox("Learning Mode (hints first)")
                if learning_mode:
                    st.info("Get hints first, then full solution!")
            
            with col2:
                category = st.selectbox("Select Analysis Type:", list(ANALYSIS_CATEGORIES.values()))
            
            st.divider()
            
            st.write("### Upload Your Data")
            uploaded_file = st.file_uploader("Choose file:", type=["jpg", "jpeg", "png", "xlsx", "xls", "csv"])
            
            df = None
            stats_results = None
            
            if uploaded_file:
                if uploaded_file.name.endswith(('jpg', 'jpeg', 'png')):
                    st.image(uploaded_file, caption="Your problem", use_container_width=True)
                else:
                    st.info(f"File: {uploaded_file.name}")
                    df = read_excel_csv(uploaded_file)
                    
                    if df is not None:
                        st.write("**Preview:**")
                        st.dataframe(df.head(), use_container_width=True)
                        
                        data_analysis_mode = st.checkbox("Analyze this data")
                        
                        if data_analysis_mode:
                            st.write("### Statistical Analysis")
                            
                            try:
                                if category == "Descriptive Statistics":
                                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                                    if numeric_cols:
                                        col_to_analyze = st.selectbox("Column:", numeric_cols)
                                        stats_dict = calculate_descriptive_stats(df, col_to_analyze)
                                        stats_df = pd.DataFrame(list(stats_dict.items()), columns=['Statistic', 'Value'])
                                        st.dataframe(stats_df, use_container_width=True)
                                        fig = create_visualizations(df, col_to_analyze)
                                        st.pyplot(fig)
                                        stats_results = str(stats_dict)
                                
                                elif category == "Hypothesis Testing":
                                    stats_results = analyze_hypothesis_testing(df)
                                elif category == "Regression":
                                    stats_results = analyze_regression(df)
                                elif category == "ANOVA":
                                    stats_results = analyze_anova(df)
                                elif category == "Machine Learning":
                                    stats_results = analyze_machine_learning(df)
                                elif category == "Predictive Modeling":
                                    stats_results = analyze_predictive_modeling(df)
                                elif category == "Clustering":
                                    stats_results = analyze_clustering(df)
                                elif category == "Correlation":
                                    stats_results = analyze_correlation(df)
                            except Exception as e:
                                st.error(f"Analysis error: {str(e)}")
            
            st.divider()
            
            st.write("### Or Type Your Problem")
            problem = st.text_area("Your problem:", placeholder="Type here...")
            
            st.warning("⚠️ Do NOT use on proctored exams!")
            
            if st.button("SOLVE & LEARN", use_container_width=True):
                problem_text_final = problem.strip() if problem else ""
                image_b64 = None
                
                if stats_results:
                    problem_text_final = f"Results: {stats_results}\n\nExplain: {problem_text_final}"
                
                if uploaded_file:
                    if uploaded_file.name.endswith(('jpg', 'jpeg', 'png')):
                        image_b64 = image_to_base64(uploaded_file)
                    elif not data_analysis_mode and df is not None:
                        problem_text_final = f"Data:\n{df.to_string()}\n\n{problem_text_final}"
                
                if problem_text_final or image_b64:
                    with st.spinner("Analyzing..."):
                        solution = solve_problem_with_ai(problem_text_final, category, current_api_key, image_b64, learning_mode)
                    
                    if solution:
                        st.divider()
                        st.subheader("✅ Solution")
                        st.markdown(solution)
                else:
                    st.error("Enter problem or upload file")
    
    elif current_page == "resources":
        st.header("📈 Resources")
        st.write("Free study materials")
        st.divider()
        
        tab1, tab2, tab3 = st.tabs(["Formulas", "Guides", "FAQ"])
        
        with tab1:
            st.write("**Common Formulas:**")
            st.markdown("""
            - Mean: $$\\mu = \\frac{\\sum x}{n}$$
            - Std Dev: $$\\sigma = \\sqrt{\\frac{\\sum(x-\\mu)^2}{n}}$$
            - t-statistic: $$t = \\frac{\\bar{x} - \\mu_0}{s/\\sqrt{n}}$$
            """)
        
        with tab2:
            st.write("**Guides Coming Soon**")
        
        with tab3:
            st.write("**Q: Can I cheat?** A: No - learning focused!")
            st.write("**Q: Use on exams?** A: NO - violates honor code!")
            st.write("**Q: Data safe?** A: Yes - instant deletion!")
    
    elif current_page == "analytics":
        st.header("Analytics")
        st.info("Coming soon!")
    
    # Admin
    if st.session_state.user_id == "admin":
        st.divider()
        st.header("ADMIN")
        st.divider()
        
        st.subheader("API Key")
        api_key = st.text_input("API Key:", type="password", value=load_api_key() or "")
        
        if api_key and save_api_key(api_key):
            st.session_state.api_key = api_key
            st.success("Saved!")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Calls", st.session_state.api_calls)
        col2.metric("Cost", f"${st.session_state.api_costs:.2f}")
        col3.metric("Margin", "80%")
