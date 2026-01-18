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
# API KEY & INITIALIZATION
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
        st.error(f"Error: {str(e)}")
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
    data_clean = data.dropna()
    shapiro_stat, shapiro_p = shapiro(data_clean)
    anderson_result = anderson(data_clean)
    ks_stat, ks_p = kstest(data_clean, 'norm', args=(data_clean.mean(), data_clean.std()))
    skewness = stats.skew(data_clean)
    kurtosis = stats.kurtosis(data_clean)
    
    return {
        "Shapiro-Wilk p-value": f"{shapiro_p:.6f}",
        "Anderson-Darling": f"{anderson_result.statistic:.6f}",
        "K-S p-value": f"{ks_p:.6f}",
        "Skewness": f"{skewness:.4f}",
        "Kurtosis": f"{kurtosis:.4f}",
        "Is Normal?": "Yes ✅" if shapiro_p > 0.05 else "No ❌"
    }

def apply_transformation(data, method):
    data_clean = data.dropna()
    
    if method == "Log":
        if (data_clean > 0).all():
            return np.log(data_clean), "Log Transformation: Compresses right-skewed data"
        else:
            st.warning("Log requires positive values!")
            return None, ""
    elif method == "Box-Cox":
        if (data_clean > 0).all():
            try:
                transformed, lam = boxcox(data_clean)
                return transformed, f"Box-Cox (λ={lam:.4f}): Auto-finds optimal transformation"
            except:
                return None, ""
        else:
            st.warning("Box-Cox needs positive values!")
            return None, ""
    elif method == "Yeo-Johnson":
        try:
            transformed, lam = yeojohnson(data_clean)
            return transformed, f"Yeo-Johnson (λ={lam:.4f}): Works with any values"
        except:
            return None, ""
    elif method == "Z-Score":
        mean = data_clean.mean()
        std = data_clean.std()
        return (data_clean - mean) / std, "Z-Score: Centers and scales data"
    
    return None, ""

def analyze_descriptive_stats(df):
    st.write("### 📊 Descriptive Statistics")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if numeric_cols:
        column = st.selectbox("Select column:", numeric_cols, key="desc_col")
        stats_dict = calculate_descriptive_stats(df, column)
        st.dataframe(pd.DataFrame(list(stats_dict.items()), columns=['Statistic', 'Value']), use_container_width=True)
        fig = create_visualizations(df, column)
        st.pyplot(fig)
        return "Stats calculated"
    return None

def analyze_normality(df):
    st.write("### ✨ Normality Testing")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if numeric_cols:
        column = st.selectbox("Select column:", numeric_cols, key="norm_col")
        data = df[column].dropna()
        
        st.write("**Normality Tests:**")
        results = test_normality(data)
        st.dataframe(pd.DataFrame(list(results.items()), columns=['Test', 'Result']), use_container_width=True)
        
        st.write("**Visualizations:**")
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
        axes[1, 1].set_title('Density')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.write("**Apply Transformation:**")
        method = st.selectbox("Method:", ["Log", "Box-Cox", "Yeo-Johnson", "Z-Score"], key="trans_method")
        transformed, explanation = apply_transformation(data, method)
        
        if transformed is not None:
            st.info(explanation)
            st.write("**Transformed Data Tests:**")
            trans_results = test_normality(pd.Series(transformed))
            st.dataframe(pd.DataFrame(list(trans_results.items()), columns=['Test', 'Result']), use_container_width=True)
            
            return "Normality analyzed"
    
    return None

def analyze_regression(df):
    st.write("### 📉 Regression Analysis")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) >= 2:
        x_col = st.selectbox("X (Independent):", numeric_cols, key="reg_x")
        y_col = st.selectbox("Y (Dependent):", numeric_cols, key="reg_y")
        
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
            "R-squared": f"{r_squared:.4f}",
            "Slope": f"{slope:.4f}",
        }
        
        st.write("**Results:**")
        st.dataframe(pd.DataFrame(list(results.items()), columns=['Metric', 'Value']), use_container_width=True)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(X, y, alpha=0.6)
        X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        y_line = model.predict(X_line)
        ax.plot(X_line, y_line, 'r-', label=f'Y = {intercept:.2f} + {slope:.2f}*X')
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.legend()
        st.pyplot(fig)
        
        return "Regression analyzed"
    
    return None

def analyze_correlation(df):
    st.write("### 📋 Correlation Analysis")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr()
        st.dataframe(corr, use_container_width=True)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
        st.pyplot(fig)
        
        return "Correlation analyzed"
    
    return None

def analyze_anova(df):
    st.write("### 📌 ANOVA Analysis")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if numeric_cols and cat_cols:
        num_col = st.selectbox("Numeric Variable:", numeric_cols, key="anova_num")
        cat_col = st.selectbox("Categorical Variable:", cat_cols, key="anova_cat")
        
        groups = [group[num_col].dropna().values for name, group in df.groupby(cat_col)]
        f_stat, p_value = f_oneway(*groups)
        
        results = {
            "F-statistic": f"{f_stat:.4f}",
            "p-value": f"{p_value:.6f}",
            "Significant": "Yes ✅" if p_value < 0.05 else "No"
        }
        
        st.write("**Results:**")
        st.dataframe(pd.DataFrame(list(results.items()), columns=['Metric', 'Value']), use_container_width=True)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        df.boxplot(column=num_col, by=cat_col, ax=ax)
        st.pyplot(fig)
        
        return "ANOVA analyzed"
    
    return None

def analyze_clustering(df):
    st.write("### 📊 Clustering Analysis")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) >= 2:
        n_clusters = st.slider("Clusters:", 2, 10, 3, key="clusters")
        
        X = df[numeric_cols].fillna(df[numeric_cols].mean())
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        results = {
            "Number of Clusters": n_clusters,
            "Inertia": f"{kmeans.inertia_:.4f}",
            "Samples": len(df)
        }
        
        st.write("**Results:**")
        st.dataframe(pd.DataFrame(list(results.items()), columns=['Metric', 'Value']), use_container_width=True)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(X[numeric_cols[0]], X[numeric_cols[1]], c=clusters, cmap='viridis', s=50)
        ax.set_xlabel(numeric_cols[0])
        ax.set_ylabel(numeric_cols[1])
        plt.colorbar(scatter, ax=ax)
        st.pyplot(fig)
        
        return "Clustering analyzed"
    
    return None

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
            st.write("**Demo Credentials:**")
            st.code("student@example.com\npassword")
            email = st.text_input("Email:")
            pwd = st.text_input("Password:", type="password")
            
            if st.button("Sign In", use_container_width=True):
                if email == "student@example.com" and pwd == "password":
                    st.session_state.user_id = f"student_{email}"
                    st.session_state.user_name = "Student"
                    st.rerun()
                else:
                    st.error("Invalid credentials")
        else:
            pwd = st.text_input("Admin Password:", type="password")
            if st.button("Sign In", use_container_width=True):
                if pwd == "admin123":
                    st.session_state.user_id = "admin"
                    st.session_state.user_name = "Admin"
                    st.rerun()
                else:
                    st.error("Invalid")
    else:
        st.write(f"### {st.session_state.user_name}! 👋")
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
# LANDING PAGE
# ============================================================================

if st.session_state.current_page == "home" and not st.session_state.user_id:
    st.title("📊 Data Insight Studio")
    st.subheader("Professional Statistics & Machine Learning Helper")
    st.divider()
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.write("""
        ### Welcome! 🎓
        
        Master statistics with AI-guided learning.
        Upload data, choose analysis, learn step-by-step!
        
        **Pricing:** $14.99/term | **Academic Focus**
        """)
    with col2:
        st.warning("⚖️ Do NOT use on exams!")
    
    st.divider()
    st.write("## 🚀 Available Tools")
    
    tools = [
        ("📊 Descriptive Statistics", "Mean, median, SD, visualizations"),
        ("✨ Normality Testing", "Test normality, transform data"),
        ("📉 Regression", "Linear regression, R², predictions"),
        ("📋 Correlation", "Heatmaps, relationships"),
        ("📌 ANOVA", "Group comparisons"),
        ("📊 Clustering", "K-Means, patterns"),
    ]
    
    cols = st.columns(2)
    for idx, (title, desc) in enumerate(tools):
        with cols[idx % 2]:
            st.write(f"### {title}\n{desc}")
    
    st.divider()
    st.success("👉 **Sign in to start analyzing!**")

# ============================================================================
# HOMEWORK HELP PAGE
# ============================================================================

elif st.session_state.current_page == "homework" and st.session_state.user_id:
    st.header("📚 Homework Help")
    st.write("Upload file, enter text, choose analysis type, get results!")
    st.divider()
    
    # Top section: Category selection
    st.write("### Select Analysis Type:")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Descriptive\nStatistics", use_container_width=True):
            st.session_state.selected_category = "Descriptive Statistics"
    with col2:
        if st.button("✨ Normality\nTesting", use_container_width=True):
            st.session_state.selected_category = "Normality Testing"
    with col3:
        if st.button("📉 Regression\nAnalysis", use_container_width=True):
            st.session_state.selected_category = "Regression Analysis"
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📋 Correlation\nAnalysis", use_container_width=True):
            st.session_state.selected_category = "Correlation Analysis"
    with col2:
        if st.button("📌 ANOVA", use_container_width=True):
            st.session_state.selected_category = "ANOVA"
    with col3:
        if st.button("📊 Clustering\nAnalysis", use_container_width=True):
            st.session_state.selected_category = "Clustering Analysis"
    
    st.divider()
    
    # File upload
    st.write("### Upload Data or Type Problem:")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Upload file:", type=["csv", "xlsx", "xls", "jpg", "jpeg", "png"])
    
    with col2:
        problem_text = st.text_area("Or type your problem:", height=100, placeholder="Type your question here...")
    
    st.divider()
    
    # Analysis section
    if "selected_category" in st.session_state:
        st.write(f"### Analysis: {st.session_state.selected_category}")
        
        df = None
        if uploaded_file:
            if uploaded_file.name.endswith(('jpg', 'jpeg', 'png')):
                st.image(uploaded_file, use_container_width=True)
            else:
                df = read_excel_csv(uploaded_file)
                if df is not None:
                    st.write("**Data Preview:**")
                    st.dataframe(df.head(), use_container_width=True)
        
        if st.button("RUN ANALYSIS 🚀", use_container_width=True):
            if df is not None:
                try:
                    if st.session_state.selected_category == "Descriptive Statistics":
                        analyze_descriptive_stats(df)
                    elif st.session_state.selected_category == "Normality Testing":
                        analyze_normality(df)
                    elif st.session_state.selected_category == "Regression Analysis":
                        analyze_regression(df)
                    elif st.session_state.selected_category == "Correlation Analysis":
                        analyze_correlation(df)
                    elif st.session_state.selected_category == "ANOVA":
                        analyze_anova(df)
                    elif st.session_state.selected_category == "Clustering Analysis":
                        analyze_clustering(df)
                    
                    st.success("✅ Analysis complete!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
            elif problem_text:
                st.info(f"Problem: {problem_text}\n\nUpload data or image for analysis")
            else:
                st.warning("Upload file or enter problem text!")
    else:
        st.info("👆 Select analysis type above to get started!")

# ============================================================================
# HOME PAGE (AFTER LOGIN)
# ============================================================================

elif st.session_state.current_page == "home" and st.session_state.user_id:
    st.title("📊 Data Insight Studio")
    st.write("Welcome! Click **Homework Help** in sidebar to analyze your data.")

# ============================================================================
# RESOURCES PAGE
# ============================================================================

elif st.session_state.current_page == "resources":
    st.header("📈 Resources")
    st.write("Coming soon!")

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
    col2.metric("Cost", "$0.00")
