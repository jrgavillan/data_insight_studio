import streamlit as st
import requests
import base64
import json
import os
from datetime import datetime
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import ttest_ind, ttest_rel, f_oneway, chi2_contingency, linregress, norm, shapiro, anderson, kstest, boxcox, yeojohnson
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, PowerTransformer, QuantileTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Data Insight Studio", layout="wide", initial_sidebar_state="expanded")

CONFIG_FILE = "api_config.txt"
PRIVACY_POLICY_URL = "https://jrgavillan.github.io/data_insight_studio/privacy_policy.md"
TERMS_URL = "https://jrgavillan.github.io/data_insight_studio/privacy_policy.md"

API_COSTS = {"per_1m_input": 0.003, "per_1m_output": 0.015, "avg_input_tokens": 500, "avg_output_tokens": 300}
PRICING = {"per_term": 14.99, "term_days": 90}

ANALYSIS_CATEGORIES = {
    "📊 DESCRIPTIVE STATISTICS": "Descriptive Statistics",
    "📈 HYPOTHESIS TESTING": "Hypothesis Testing",
    "📉 REGRESSION ANALYSIS": "Regression",
    "📌 ANOVA": "ANOVA",
    "📊 CONFIDENCE INTERVALS": "Confidence Intervals",
    "🔔 PROBABILITY DISTRIBUTIONS": "Probability",
    "🧪 T-TESTS": "T-Tests",
    "🎯 CHI-SQUARE TEST": "Chi-Square",
    "✨ NORMALITY TESTING & TRANSFORMATIONS": "Normality Testing",
    "🤖 MACHINE LEARNING": "Machine Learning",
    "🔮 PREDICTIVE MODELING": "Predictive Modeling",
    "📊 CLUSTERING ANALYSIS": "Clustering",
    "📋 CORRELATION ANALYSIS": "Correlation",
}

# ============================================================================
# API KEY MANAGEMENT
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
# NORMALITY TESTING & TRANSFORMATIONS
# ============================================================================

def test_normality(data):
    """Comprehensive normality testing"""
    data_clean = data.dropna()
    
    # Shapiro-Wilk Test
    shapiro_stat, shapiro_p = shapiro(data_clean)
    
    # Anderson-Darling Test
    anderson_result = anderson(data_clean)
    anderson_stat = anderson_result.statistic
    
    # Kolmogorov-Smirnov Test
    ks_stat, ks_p = kstest(data_clean, 'norm', args=(data_clean.mean(), data_clean.std()))
    
    # Skewness and Kurtosis
    skewness = stats.skew(data_clean)
    kurtosis = stats.kurtosis(data_clean)
    
    results = {
        "Shapiro-Wilk Statistic": shapiro_stat,
        "Shapiro-Wilk p-value": shapiro_p,
        "Anderson-Darling Statistic": anderson_stat,
        "K-S Statistic": ks_stat,
        "K-S p-value": ks_p,
        "Skewness": skewness,
        "Kurtosis": kurtosis,
        "Normal Distribution": "Yes (p > 0.05)" if shapiro_p > 0.05 else "No (p < 0.05)"
    }
    
    return results

def apply_transformation(data, method):
    """Apply various transformations to normalize data"""
    data_clean = data.dropna()
    transformed_data = None
    explanation = ""
    
    if method == "Log Transformation":
        # Only works with positive values
        if (data_clean > 0).all():
            transformed_data = np.log(data_clean)
            explanation = "Log transformation compresses right-skewed data. Works only with positive values. Good for multiplicative relationships."
        else:
            st.warning("Log transformation requires all positive values!")
            return None, ""
    
    elif method == "Square Root Transformation":
        if (data_clean >= 0).all():
            transformed_data = np.sqrt(data_clean)
            explanation = "Square root transformation is less aggressive than log. Good for moderate skewness and count data."
        else:
            st.warning("Square root transformation requires non-negative values!")
            return None, ""
    
    elif method == "Box-Cox Transformation":
        if (data_clean > 0).all():
            try:
                transformed_data, lambda_param = boxcox(data_clean)
                explanation = f"Box-Cox transformation optimally transforms data (λ={lambda_param:.4f}). Automatically finds best transformation parameter."
            except:
                st.warning("Box-Cox transformation failed!")
                return None, ""
        else:
            st.warning("Box-Cox requires positive values!")
            return None, ""
    
    elif method == "Yeo-Johnson Transformation":
        try:
            transformed_data, lambda_param = yeojohnson(data_clean)
            explanation = f"Yeo-Johnson transformation (λ={lambda_param:.4f}). Works with any values including negative/zero. More flexible than Box-Cox."
        except:
            st.warning("Yeo-Johnson transformation failed!")
            return None, ""
    
    elif method == "Z-Score Standardization":
        mean = data_clean.mean()
        std = data_clean.std()
        transformed_data = (data_clean - mean) / std
        explanation = "Z-score standardization centers data (mean=0) and scales by std dev (std=1). Preserves distribution shape."
    
    elif method == "Min-Max Normalization":
        min_val = data_clean.min()
        max_val = data_clean.max()
        transformed_data = (data_clean - min_val) / (max_val - min_val)
        explanation = "Min-Max scaling transforms data to [0,1] range. Preserves relationships and distribution shape."
    
    return transformed_data, explanation

def analyze_normality_testing(df):
    """Full normality testing and transformation analysis"""
    st.write("### Normality Testing & Transformations Analysis")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 1:
        st.warning("Need at least 1 numeric column")
        return None
    
    column = st.selectbox("Select Column to Analyze:", numeric_cols, key="norm_col")
    data = df[column].dropna()
    
    # ===== NORMALITY TESTING =====
    st.write("#### 1️⃣ NORMALITY TESTING")
    
    normality_results = test_normality(data)
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Test Results:**")
        results_df = pd.DataFrame(list(normality_results.items()), columns=['Test', 'Result'])
        st.dataframe(results_df, use_container_width=True)
    
    with col2:
        st.write("**Interpretation:**")
        if normality_results["Shapiro-Wilk p-value"] > 0.05:
            st.success("✅ Data appears normally distributed (p > 0.05)")
        else:
            st.error("❌ Data is NOT normally distributed (p < 0.05)")
        
        if abs(normality_results["Skewness"]) > 2:
            st.warning(f"⚠️ High skewness ({normality_results['Skewness']:.2f}) - transformation recommended")
        if abs(normality_results["Kurtosis"]) > 3:
            st.warning(f"⚠️ High kurtosis ({normality_results['Kurtosis']:.2f}) - heavy/light tails")
    
    # ===== VISUALIZATIONS =====
    st.write("#### 2️⃣ ORIGINAL DATA VISUALIZATION")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Histogram
    axes[0, 0].hist(data, bins=30, edgecolor='black', alpha=0.7, density=True)
    mu, sigma = norm.fit(data)
    x = np.linspace(data.min(), data.max(), 100)
    axes[0, 0].plot(x, norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal Fit')
    axes[0, 0].set_title('Histogram with Normal Curve')
    axes[0, 0].legend()
    
    # Q-Q Plot
    stats.probplot(data, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Q-Q Plot')
    
    # Box Plot
    axes[1, 0].boxplot(data)
    axes[1, 0].set_title('Box Plot')
    axes[1, 0].set_ylabel(column)
    
    # Density Plot
    data.plot(kind='density', ax=axes[1, 1])
    axes[1, 1].set_title('Density Plot')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # ===== TRANSFORMATIONS =====
    st.write("#### 3️⃣ DATA TRANSFORMATIONS")
    
    transformation_method = st.selectbox(
        "Choose Transformation Method:",
        ["Log Transformation", "Square Root Transformation", "Box-Cox Transformation", 
         "Yeo-Johnson Transformation", "Z-Score Standardization", "Min-Max Normalization"],
        key="transform_method"
    )
    
    transformed_data, explanation = apply_transformation(data, transformation_method)
    
    if transformed_data is not None:
        st.write(f"**Why {transformation_method}?**")
        st.info(explanation)
        
        # ===== AFTER TRANSFORMATION =====
        st.write("#### 4️⃣ TRANSFORMED DATA ANALYSIS")
        
        transformed_results = test_normality(pd.Series(transformed_data))
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Transformed Data Tests:**")
            trans_df = pd.DataFrame(list(transformed_results.items()), columns=['Test', 'Result'])
            st.dataframe(trans_df, use_container_width=True)
        
        with col2:
            st.write("**Improvement Check:**")
            if transformed_results["Shapiro-Wilk p-value"] > normality_results["Shapiro-Wilk p-value"]:
                improvement = ((transformed_results["Shapiro-Wilk p-value"] - normality_results["Shapiro-Wilk p-value"]) / 
                              normality_results["Shapiro-Wilk p-value"] * 100)
                st.success(f"✅ Improved by {improvement:.1f}%")
            else:
                st.info("Transformation didn't significantly improve normality")
        
        # ===== BEFORE/AFTER VISUALIZATION =====
        st.write("#### 5️⃣ BEFORE & AFTER COMPARISON")
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original
        axes[0, 0].hist(data, bins=30, edgecolor='black', alpha=0.7)
        axes[0, 0].set_title('Original: Histogram')
        axes[0, 0].set_ylabel('Frequency')
        
        stats.probplot(data, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Original: Q-Q Plot')
        
        data.plot(kind='density', ax=axes[0, 2])
        axes[0, 2].set_title('Original: Density')
        
        # Transformed
        axes[1, 0].hist(transformed_data, bins=30, edgecolor='black', alpha=0.7)
        axes[1, 0].set_title(f'Transformed: Histogram')
        axes[1, 0].set_ylabel('Frequency')
        
        stats.probplot(pd.Series(transformed_data), dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Transformed: Q-Q Plot')
        
        pd.Series(transformed_data).plot(kind='density', ax=axes[1, 2])
        axes[1, 2].set_title('Transformed: Density')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # ===== SAMPLING FEATURE =====
        st.write("#### 6️⃣ SAMPLING FOR NORMALITY")
        
        col1, col2 = st.columns(2)
        
        with col1:
            apply_sampling = st.checkbox("Apply sampling to make data approximately normal?")
        with col2:
            sample_size = st.slider("Sample Size (%):", 10, 100, 100, key="sample_size")
        
        if apply_sampling and sample_size < 100:
            sample_indices = np.random.choice(len(transformed_data), 
                                             size=int(len(transformed_data) * sample_size / 100), 
                                             replace=False)
            sampled_data = transformed_data[sample_indices]
            
            sampled_results = test_normality(pd.Series(sampled_data))
            
            st.write(f"**Sampled Data ({sample_size}%) Normality Test:**")
            sampled_df = pd.DataFrame(list(sampled_results.items()), columns=['Test', 'Result'])
            st.dataframe(sampled_df, use_container_width=True)
            
            if sampled_results["Shapiro-Wilk p-value"] > 0.05:
                st.success("✅ Sampled data is approximately normally distributed!")
            else:
                st.info("Sampled data still shows deviation from normality")
            
            # Create sampled dataset
            updated_df = df.copy()
            updated_df[column] = np.nan
            updated_df.loc[df.index[sample_indices], column] = sampled_data
            updated_df = updated_df.dropna()
            
            st.write("**Updated Dataset (Sampled):**")
            st.dataframe(updated_df.head(10), use_container_width=True)
            st.download_button(
                label="Download Sampled Dataset",
                data=updated_df.to_csv(index=False),
                file_name=f"sampled_data_{column}.csv",
                mime="text/csv"
            )
            
            return str(sampled_results)
        
        return str(transformed_results)
    
    return None

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def image_to_base64(image_file):
    return base64.b64encode(image_file.read()).decode()

def read_excel_csv(file):
    try:
        if file.name.endswith('.csv'):
            return pd.read_csv(file)
        elif file.name.endswith(('.xlsx', '.xls')):
            return pd.read_excel(file)
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
    
    results = {
        "Equation": f"Y = {intercept:.4f} + {slope:.4f}*X",
        "R-squared": r_squared,
        "Slope": slope,
        "Intercept": intercept,
    }
    
    st.write("**Results:**")
    results_df = pd.DataFrame(list(results.items()), columns=['Metric', 'Value'])
    st.dataframe(results_df, use_container_width=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X, y, alpha=0.6)
    X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_line = model.predict(X_line)
    ax.plot(X_line, y_line, 'r-', label=f'Fit: Y = {intercept:.2f} + {slope:.2f}*X')
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title('Linear Regression')
    ax.legend()
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
            st.error("Please enter a problem")
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
        
        system_prompt = f"""You are an expert statistics tutor. Help students UNDERSTAND concepts.

IMPORTANT RULES:
1. Show step-by-step work
2. Explain the WHY
3. Use LaTeX for formulas: $$formula$$
4. Never just give answers

Category: {category}"""
        
        text_prompt = f"""Analyze and explain this step-by-step.

PROBLEM: {problem_text}"""
        
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
            
            st.session_state.api_calls += 1
            st.session_state.api_costs += estimate_problem_cost()
            
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
if "current_page" not in st.session_state:
    st.session_state.current_page = "home"

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
            st.write("**Demo:** student@example.com / password")
            student_email = st.text_input("Email:", key="student_email")
            student_pass = st.text_input("Password:", type="password", key="student_pass")
            terms_check = st.checkbox("I agree to Terms", key="terms_agree")
            
            if st.button("Sign In", key="student_signin"):
                if not terms_check:
                    st.error("Accept terms")
                elif student_email == "student@example.com" and student_pass == "password":
                    st.session_state.user_id = f"student_{student_email}"
                    st.session_state.user_name = "Student"
                    st.rerun()
                else:
                    st.error("Invalid credentials")
        else:
            st.write("**Admin Password**")
            admin_pass = st.text_input("Password:", type="password", key="admin_pass")
            
            if st.button("Sign In", key="admin_signin"):
                if admin_pass == "admin123":
                    st.session_state.user_id = "admin"
                    st.session_state.user_name = "Admin"
                    st.rerun()
                else:
                    st.error("Invalid")
        
        st.divider()
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
    st.error("Please sign in")
else:
    current_page = st.session_state.current_page
    
    if current_page == "home":
        st.title("📊 Data Insight Studio")
        st.subheader("Professional Statistics & Machine Learning Homework Helper")
        st.divider()
        
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
            st.info("⚖️ **Academic Integrity**\n\nDo NOT use on exams!")
        
        st.divider()
        
        st.write("## 🚀 Available Analysis & Tools")
        st.write("Click **Homework Help** to get started with any of these:")
        
        cols = st.columns(2)
        analysis_list = list(ANALYSIS_CATEGORIES.items())
        
        for idx, (display_name, category) in enumerate(analysis_list):
            with cols[idx % 2]:
                st.write(f"### {display_name}")
                if category == "Normality Testing":
                    st.write("✨ Test for normality, transformations, sampling for approximate normality")
                elif category == "Descriptive Statistics":
                    st.write("📈 Mean, median, std dev, visualizations")
                elif category == "Machine Learning":
                    st.write("🤖 Random Forest, Gradient Boosting, features")
                elif category == "Correlation":
                    st.write("📋 Correlation matrix, heatmaps")
                else:
                    st.write("Statistical analysis and testing")
        
        st.divider()
        st.success("Ready? Click **Homework Help** in the sidebar! 🚀")
    
    elif current_page == "homework":
        st.header("📚 Homework Help")
        st.write("Upload data or image, select analysis type!")
        st.divider()
        
        current_api_key = get_api_key()
        
        if not current_api_key:
            st.warning("System not configured")
        else:
            col1, col2 = st.columns(2)
            with col1:
                learning_mode = st.checkbox("Learning Mode")
            with col2:
                category = st.selectbox("Select Analysis:", list(ANALYSIS_CATEGORIES.values()))
            
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
                            st.write("### Analysis")
                            
                            try:
                                if category == "Normality Testing":
                                    stats_results = analyze_normality_testing(df)
                                elif category == "Descriptive Statistics":
                                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                                    if numeric_cols:
                                        col_to_analyze = st.selectbox("Column:", numeric_cols)
                                        stats_dict = calculate_descriptive_stats(df, col_to_analyze)
                                        stats_df = pd.DataFrame(list(stats_dict.items()), columns=['Statistic', 'Value'])
                                        st.dataframe(stats_df, use_container_width=True)
                                        fig = create_visualizations(df, col_to_analyze)
                                        st.pyplot(fig)
                                        stats_results = str(stats_dict)
                                elif category == "Regression":
                                    stats_results = analyze_regression(df)
                                elif category == "Correlation":
                                    stats_results = analyze_correlation(df)
                            except Exception as e:
                                st.error(f"Error: {str(e)}")
            
            st.divider()
            st.write("### Or Type Your Problem")
            problem = st.text_area("Your problem:", placeholder="Type here...")
            
            st.warning("⚠️ Do NOT use on proctored exams!")
            
            if st.button("SOLVE & LEARN", use_container_width=True):
                problem_text_final = problem.strip() if problem else ""
                
                if stats_results:
                    problem_text_final = f"Results: {stats_results}\n\nExplain: {problem_text_final}"
                
                if problem_text_final:
                    with st.spinner("Analyzing..."):
                        solution = solve_problem_with_ai(problem_text_final, category, current_api_key, None, learning_mode)
                    
                    if solution:
                        st.divider()
                        st.subheader("✅ Solution")
                        st.markdown(solution)
    
    elif current_page == "resources":
        st.header("📈 Resources")
        st.write("Free study materials coming soon!")
    
    elif current_page == "analytics":
        st.header("Analytics")
        st.info("Coming soon!")
    
    # Admin
    if st.session_state.user_id == "admin":
        st.divider()
        st.header("ADMIN")
        api_key = st.text_input("API Key:", type="password", value=load_api_key() or "")
        if api_key and save_api_key(api_key):
            st.session_state.api_key = api_key
            st.success("Saved!")
        
        col1, col2 = st.columns(2)
        col1.metric("API Calls", st.session_state.api_calls)
        col2.metric("Cost", f"${st.session_state.api_costs:.2f}")
