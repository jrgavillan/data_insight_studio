import streamlit as st
import requests
import base64
import os
from datetime import datetime
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import ttest_ind, f_oneway, chi2_contingency, shapiro, anderson, kstest, boxcox, yeojohnson
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Data Insight Studio", layout="wide", initial_sidebar_state="expanded")

CONFIG_FILE = "api_config.txt"

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
# INITIALIZATION
# ============================================================================

if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "user_name" not in st.session_state:
    st.session_state.user_name = None
if "api_key" not in st.session_state:
    st.session_state.api_key = None
if "current_page" not in st.session_state:
    st.session_state.current_page = "home"
if "api_calls" not in st.session_state:
    st.session_state.api_calls = 0
if "api_costs" not in st.session_state:
    st.session_state.api_costs = 0.0

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

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

def test_normality(data):
    """Comprehensive normality testing"""
    data_clean = data.dropna()
    shapiro_stat, shapiro_p = shapiro(data_clean)
    anderson_result = anderson(data_clean)
    ks_stat, ks_p = kstest(data_clean, 'norm', args=(data_clean.mean(), data_clean.std()))
    skewness = stats.skew(data_clean)
    kurtosis = stats.kurtosis(data_clean)
    
    results = {
        "Shapiro-Wilk p-value": shapiro_p,
        "Anderson-Darling Stat": anderson_result.statistic,
        "K-S p-value": ks_p,
        "Skewness": skewness,
        "Kurtosis": kurtosis,
        "Is Normal?": "Yes" if shapiro_p > 0.05 else "No"
    }
    return results

def apply_transformation(data, method):
    """Apply transformation"""
    data_clean = data.dropna()
    
    if method == "Log":
        if (data_clean > 0).all():
            return np.log(data_clean), "Log: Compresses right-skewed data. Works with positive values only."
        else:
            st.warning("Log requires positive values!")
            return None, ""
    elif method == "Square Root":
        if (data_clean >= 0).all():
            return np.sqrt(data_clean), "Square Root: Less aggressive than log. Good for moderate skewness."
        else:
            st.warning("Requires non-negative values!")
            return None, ""
    elif method == "Box-Cox":
        if (data_clean > 0).all():
            try:
                transformed, lam = boxcox(data_clean)
                return transformed, f"Box-Cox (λ={lam:.4f}): Automatically finds optimal transformation."
            except:
                st.warning("Box-Cox failed!")
                return None, ""
        else:
            st.warning("Box-Cox requires positive values!")
            return None, ""
    elif method == "Yeo-Johnson":
        try:
            transformed, lam = yeojohnson(data_clean)
            return transformed, f"Yeo-Johnson (λ={lam:.4f}): Works with any values."
        except:
            st.warning("Yeo-Johnson failed!")
            return None, ""
    elif method == "Z-Score":
        mean = data_clean.mean()
        std = data_clean.std()
        return (data_clean - mean) / std, "Z-Score: Centers (mean=0) and scales (std=1)."
    elif method == "Min-Max":
        min_val = data_clean.min()
        max_val = data_clean.max()
        return (data_clean - min_val) / (max_val - min_val), "Min-Max: Scales data to [0,1]."
    
    return None, ""

def analyze_descriptive_stats(df):
    st.write("### 📊 Descriptive Statistics")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        st.warning("Need numeric columns")
        return None
    
    column = st.selectbox("Select Column:", numeric_cols, key="desc_col")
    stats_dict = calculate_descriptive_stats(df, column)
    
    st.write("**Statistics:**")
    st.dataframe(pd.DataFrame(list(stats_dict.items()), columns=['Statistic', 'Value']), use_container_width=True)
    
    st.write("**Visualizations:**")
    fig = create_visualizations(df, column)
    st.pyplot(fig)
    
    return str(stats_dict)

def analyze_normality_testing(df):
    st.write("### ✨ Normality Testing & Transformations")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        st.warning("Need numeric columns")
        return None
    
    column = st.selectbox("Select Column:", numeric_cols, key="norm_col")
    data = df[column].dropna()
    
    # Test normality
    st.write("#### 1️⃣ Normality Tests")
    results = test_normality(data)
    st.dataframe(pd.DataFrame(list(results.items()), columns=['Test', 'Result']), use_container_width=True)
    
    # Visualization
    st.write("#### 2️⃣ Original Data")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0, 0].hist(data, bins=30, edgecolor='black', alpha=0.7, density=True)
    mu, sigma = stats.norm.fit(data)
    x = np.linspace(data.min(), data.max(), 100)
    axes[0, 0].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2)
    axes[0, 0].set_title('Histogram')
    
    stats.probplot(data, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Q-Q Plot')
    
    axes[1, 0].boxplot(data)
    axes[1, 0].set_title('Box Plot')
    
    data.plot(kind='density', ax=axes[1, 1])
    axes[1, 1].set_title('Density Plot')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Transformations
    st.write("#### 3️⃣ Apply Transformation")
    method = st.selectbox("Method:", ["Log", "Square Root", "Box-Cox", "Yeo-Johnson", "Z-Score", "Min-Max"])
    
    transformed, explanation = apply_transformation(data, method)
    if transformed is not None:
        st.info(explanation)
        
        st.write("#### 4️⃣ Transformed Data")
        trans_results = test_normality(pd.Series(transformed))
        st.dataframe(pd.DataFrame(list(trans_results.items()), columns=['Test', 'Result']), use_container_width=True)
        
        st.write("#### 5️⃣ Before & After")
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        axes[0, 0].hist(data, bins=30, edgecolor='black', alpha=0.7)
        axes[0, 0].set_title('Original')
        stats.probplot(data, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Original Q-Q')
        data.plot(kind='density', ax=axes[0, 2])
        axes[0, 2].set_title('Original Density')
        
        axes[1, 0].hist(transformed, bins=30, edgecolor='black', alpha=0.7)
        axes[1, 0].set_title('Transformed')
        stats.probplot(pd.Series(transformed), dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Transformed Q-Q')
        pd.Series(transformed).plot(kind='density', ax=axes[1, 2])
        axes[1, 2].set_title('Transformed Density')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        return str(trans_results)
    
    return None

def analyze_regression(df):
    st.write("### 📉 Regression Analysis")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.warning("Need at least 2 numeric columns")
        return None
    
    x_col = st.selectbox("X (Independent):", numeric_cols, key="reg_x")
    y_col = st.selectbox("Y (Dependent):", numeric_cols, key="reg_y", index=1 if len(numeric_cols) > 1 else 0)
    
    data_clean = df[[x_col, y_col]].dropna()
    X = data_clean[x_col].values.reshape(-1, 1)
    y = data_clean[y_col].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    r_squared = model.score(X, y)
    slope = model.coef_[0]
    intercept = model.intercept_
    
    results = {
        "Equation": f"Y = {intercept:.4f} + {slope:.4f}*X",
        "R-squared": r_squared,
        "Slope": slope,
        "Intercept": intercept,
    }
    
    st.write("**Results:**")
    st.dataframe(pd.DataFrame(list(results.items()), columns=['Metric', 'Value']), use_container_width=True)
    
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

def analyze_correlation(df):
    st.write("### 📋 Correlation Analysis")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.warning("Need at least 2 numeric columns")
        return None
    
    corr = df[numeric_cols].corr()
    st.write("**Correlation Matrix:**")
    st.dataframe(corr, use_container_width=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
    ax.set_title('Correlation Heatmap')
    st.pyplot(fig)
    
    return str(corr)

def analyze_anova(df):
    st.write("### 📌 ANOVA Analysis")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if len(numeric_cols) < 1 or len(cat_cols) < 1:
        st.warning("Need numeric and categorical columns")
        return None
    
    num_col = st.selectbox("Numeric Variable:", numeric_cols, key="anova_num")
    cat_col = st.selectbox("Categorical Variable:", cat_cols, key="anova_cat")
    
    groups = [group[num_col].dropna().values for name, group in df.groupby(cat_col)]
    f_stat, p_value = f_oneway(*groups)
    
    results = {
        "F-statistic": f_stat,
        "p-value": p_value,
        "Significant": "Yes" if p_value < 0.05 else "No"
    }
    
    st.write("**Results:**")
    st.dataframe(pd.DataFrame(list(results.items()), columns=['Metric', 'Value']), use_container_width=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    df.boxplot(column=num_col, by=cat_col, ax=ax)
    st.pyplot(fig)
    
    return str(results)

def analyze_clustering(df):
    st.write("### 📊 Clustering Analysis")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.warning("Need at least 2 numeric columns")
        return None
    
    n_clusters = st.slider("Number of Clusters:", 2, 10, 3, key="clusters")
    
    X = df[numeric_cols].fillna(df[numeric_cols].mean())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    results = {
        "Number of Clusters": n_clusters,
        "Inertia": kmeans.inertia_,
        "Samples": len(df)
    }
    
    st.write("**Results:**")
    st.dataframe(pd.DataFrame(list(results.items()), columns=['Metric', 'Value']), use_container_width=True)
    
    if len(numeric_cols) >= 2:
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(X[numeric_cols[0]], X[numeric_cols[1]], c=clusters, cmap='viridis', s=50)
        ax.set_xlabel(numeric_cols[0])
        ax.set_ylabel(numeric_cols[1])
        ax.set_title(f'K-Means Clustering (k={n_clusters})')
        plt.colorbar(scatter, ax=ax)
        st.pyplot(fig)
    
    return str(results)

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    try:
        st.image("logo_1.png", width=250)
    except:
        st.title("📊 Data Insight Studio")
    st.divider()
    
    if not st.session_state.user_id:
        st.write("### Login")
        login_type = st.radio("Select:", ["Student", "Admin"], key="login_type")
        
        if login_type == "Student":
            st.write("Demo: student@example.com / password")
            email = st.text_input("Email:", key="email_input")
            pwd = st.text_input("Password:", type="password", key="pwd_input")
            
            if st.button("Sign In", use_container_width=True):
                if email == "student@example.com" and pwd == "password":
                    st.session_state.user_id = f"student_{email}"
                    st.session_state.user_name = "Student"
                    st.rerun()
                else:
                    st.error("Invalid")
        else:
            pwd = st.text_input("Password:", type="password", key="admin_pwd")
            if st.button("Sign In", use_container_width=True):
                if pwd == "admin123":
                    st.session_state.user_id = "admin"
                    st.session_state.user_name = "Admin"
                    st.rerun()
                else:
                    st.error("Invalid")
    else:
        st.write(f"### Hi {st.session_state.user_name}! 👋")
        st.divider()
        
        if st.button("📊 Home", use_container_width=True):
            st.session_state.current_page = "home"
            st.rerun()
        if st.button("📚 Homework Help", use_container_width=True):
            st.session_state.current_page = "homework"
            st.rerun()
        if st.button("📈 Resources", use_container_width=True):
            st.session_state.current_page = "resources"
            st.rerun()
        
        st.divider()
        if st.button("🚪 Sign Out", use_container_width=True):
            st.session_state.user_id = None
            st.session_state.user_name = None
            st.rerun()

# ============================================================================
# MAIN CONTENT - LANDING PAGE (BEFORE LOGIN)
# ============================================================================

if st.session_state.current_page == "home" and not st.session_state.user_id:
    st.title("📊 Data Insight Studio")
    st.subheader("Professional Statistics & Machine Learning Homework Helper")
    st.divider()
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.write("""
        ### Welcome! 🎓
        
        Master statistics and machine learning with AI-powered guidance.
        Upload your data, choose your analysis, and learn step-by-step!
        
        **What We Offer:**
        - 14+ Statistical Analysis Types
        - Machine Learning Models
        - Normality Testing & Transformations
        - Data Visualization
        
        **Pricing:** $14.99 per 90-day term
        """)
    with col2:
        st.info("⚖️ **Academic Integrity**\n\nDo NOT use on exams!")
    
    st.divider()
    st.write("## 🚀 Available Analysis Tools")
    
    tools = [
        ("📊 Descriptive Statistics", "Mean, median, std dev, visualizations"),
        ("📈 Hypothesis Testing", "t-tests, chi-square, p-values"),
        ("📉 Regression Analysis", "Linear regression, R², predictions"),
        ("📌 ANOVA", "Group comparisons, F-statistic"),
        ("✨ Normality Testing", "Test for normality, transformations, sampling"),
        ("🤖 Machine Learning", "Random Forest, Gradient Boosting"),
        ("🔮 Predictive Modeling", "Train/test splits, accuracy"),
        ("📊 Clustering", "K-Means analysis, patterns"),
        ("📋 Correlation", "Correlation matrix, heatmaps"),
        ("🎯 Confidence Intervals", "CI ranges, margin of error"),
        ("🔔 Probability", "Distribution fitting, tests"),
        ("🧪 T-Tests", "One/two-sample, paired tests"),
    ]
    
    cols = st.columns(2)
    for idx, (title, desc) in enumerate(tools):
        with cols[idx % 2]:
            st.write(f"### {title}")
            st.write(desc)
    
    st.divider()
    st.write("## ✨ Features")
    
    fcols = st.columns(3)
    with fcols[0]:
        st.write("### 🧠 Smart Analysis")
        st.write("Auto-detect tests, instant calculations, visualizations")
    with fcols[1]:
        st.write("### 📚 Learn with AI")
        st.write("Step-by-step explanations, learning mode, plain language")
    with fcols[2]:
        st.write("### 🎯 Complete Toolkit")
        st.write("12+ analysis types, ML algorithms, data prep")
    
    st.divider()
    st.success("## 👉 Sign in above to get started! →")

# ============================================================================
# MAIN CONTENT - AFTER LOGIN
# ============================================================================

elif st.session_state.user_id:
    current_page = st.session_state.current_page
    
    if current_page == "home":
        st.title("📊 Data Insight Studio")
        st.write("Welcome! Select **Homework Help** in the sidebar to analyze your data.")
        st.divider()
        
        st.write("### Quick Start")
        st.write("1. Click **Homework Help**")
        st.write("2. Upload your CSV/Excel file or image")
        st.write("3. Select analysis type")
        st.write("4. Get instant results & explanations!")
    
    elif current_page == "homework":
        st.header("📚 Homework Help")
        st.write("Choose analysis type, upload data, get results!")
        st.divider()
        
        # Select analysis type FIRST
        st.write("### Step 1: Select Analysis Type")
        category = st.selectbox("Choose Analysis:", [
            "Descriptive Statistics",
            "Normality Testing & Transformations",
            "Regression Analysis",
            "Correlation Analysis",
            "ANOVA",
            "Clustering Analysis"
        ], key="main_category")
        
        st.divider()
        
        # Upload file
        st.write("### Step 2: Upload Data")
        uploaded_file = st.file_uploader("Choose file:", type=["csv", "xlsx", "xls", "jpg", "jpeg", "png"])
        
        if uploaded_file:
            if uploaded_file.name.endswith(('jpg', 'jpeg', 'png')):
                st.image(uploaded_file, use_container_width=True)
                st.write("Image uploaded. Analyze manually or ask AI for help.")
            else:
                df = read_excel_csv(uploaded_file)
                if df is not None:
                    st.write("**Data Preview:**")
                    st.dataframe(df.head(), use_container_width=True)
                    
                    st.divider()
                    st.write("### Step 3: Analyze")
                    
                    if st.checkbox("Run Analysis"):
                        try:
                            if category == "Descriptive Statistics":
                                analyze_descriptive_stats(df)
                            elif category == "Normality Testing & Transformations":
                                analyze_normality_testing(df)
                            elif category == "Regression Analysis":
                                analyze_regression(df)
                            elif category == "Correlation Analysis":
                                analyze_correlation(df)
                            elif category == "ANOVA":
                                analyze_anova(df)
                            elif category == "Clustering Analysis":
                                analyze_clustering(df)
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
    
    elif current_page == "resources":
        st.header("📈 Resources")
        st.write("Free study materials coming soon!")

# ============================================================================
# ADMIN PANEL
# ============================================================================

if st.session_state.user_id == "admin":
    st.divider()
    st.header("⚙️ Admin Panel")
    api_key = st.text_input("API Key:", type="password", value=load_api_key() or "")
    if api_key and save_api_key(api_key):
        st.session_state.api_key = api_key
        st.success("✅ Saved!")
    
    col1, col2 = st.columns(2)
    col1.metric("API Calls", st.session_state.api_calls)
    col2.metric("Cost", f"${st.session_state.api_costs:.2f}")
