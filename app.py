import streamlit as st
import requests
import os
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import ttest_ind, f_oneway, shapiro, anderson, kstest, boxcox, yeojohnson
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Data Insight Studio", layout="wide", initial_sidebar_state="expanded")

CONFIG_FILE = "api_config.txt"

# ============================================================================
# SESSION STATE
# ============================================================================

if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "user_name" not in st.session_state:
    st.session_state.user_name = None
if "api_key" not in st.session_state:
    st.session_state.api_key = None
if "current_page" not in st.session_state:
    st.session_state.current_page = "home"

# ============================================================================
# API KEY MANAGEMENT
# ============================================================================

def save_api_key(key):
    try:
        with open(CONFIG_FILE, "w") as f:
            f.write(key.strip())
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
                key = f.read().strip()
                if key:
                    return key
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
# AI SOLVER
# ============================================================================

def solve_with_ai(problem_text, category, api_key):
    try:
        if not api_key:
            return "❌ ERROR: API key not configured by admin"
        
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
                "system": f"You are an expert statistics tutor. Help students understand. Category: {category}",
                "messages": [{"role": "user", "content": problem_text}]
            },
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()['content'][0]['text']
        else:
            return f"❌ API Error: {response.status_code}"
    except Exception as e:
        return f"❌ Error: {str(e)}"

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def read_file(file):
    try:
        if file.name.endswith('.csv'):
            return pd.read_csv(file)
        elif file.name.endswith(('.xlsx', '.xls')):
            return pd.read_excel(file)
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return None

def descriptive_stats(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        st.error("No numeric columns found")
        return
    
    col = st.selectbox("Choose column:", numeric_cols, key="desc_col")
    
    stats_dict = {
        "Mean": f"{df[col].mean():.4f}",
        "Median": f"{df[col].median():.4f}",
        "Std Dev": f"{df[col].std():.4f}",
        "Min": f"{df[col].min():.4f}",
        "Max": f"{df[col].max():.4f}",
        "N": len(df)
    }
    
    st.dataframe(pd.DataFrame(list(stats_dict.items()), columns=['Statistic', 'Value']), use_container_width=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0, 0].hist(df[col].dropna(), bins=30, edgecolor='black')
    axes[0, 0].set_title('Histogram')
    axes[0, 1].boxplot(df[col].dropna())
    axes[0, 1].set_title('Box Plot')
    stats.probplot(df[col].dropna(), dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot')
    df[col].plot(kind='density', ax=axes[1, 1])
    axes[1, 1].set_title('Density')
    plt.tight_layout()
    st.pyplot(fig)

def normality_test(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        st.error("No numeric columns found")
        return
    
    col = st.selectbox("Choose column:", numeric_cols, key="norm_col")
    data = df[col].dropna()
    
    shapiro_stat, shapiro_p = shapiro(data)
    
    results = {
        "Shapiro-Wilk p-value": f"{shapiro_p:.6f}",
        "Skewness": f"{stats.skew(data):.4f}",
        "Is Normal?": "Yes ✅" if shapiro_p > 0.05 else "No ❌"
    }
    
    st.dataframe(pd.DataFrame(list(results.items()), columns=['Test', 'Result']), use_container_width=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0, 0].hist(data, bins=30, edgecolor='black')
    axes[0, 0].set_title('Original')
    stats.probplot(data, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Q-Q Original')
    axes[1, 0].text(0.5, 0.5, 'Transformations:', ha='center')
    axes[1, 1].text(0.5, 0.5, 'Try Log, Box-Cox,\nYeo-Johnson', ha='center')
    plt.tight_layout()
    st.pyplot(fig)

def regression_analysis(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        st.error("Need at least 2 numeric columns")
        return
    
    col1, col2 = st.columns(2)
    with col1:
        x_col = st.selectbox("X (Independent):", numeric_cols, key="x_col")
    with col2:
        y_col = st.selectbox("Y (Dependent):", numeric_cols, key="y_col", index=1)
    
    data_clean = df[[x_col, y_col]].dropna()
    X = data_clean[x_col].values.reshape(-1, 1)
    y = data_clean[y_col].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    r_sq = model.score(X, y)
    slope = model.coef_[0]
    intercept = model.intercept_
    
    results = {
        "Equation": f"Y = {intercept:.4f} + {slope:.4f}*X",
        "R²": f"{r_sq:.4f}",
        "Slope": f"{slope:.4f}"
    }
    
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

def correlation_analysis(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        st.error("Need at least 2 numeric columns")
        return
    
    corr = df[numeric_cols].corr()
    st.dataframe(corr, use_container_width=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
    st.pyplot(fig)

def anova_analysis(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if not numeric_cols or not cat_cols:
        st.error("Need numeric and categorical columns")
        return
    
    col1, col2 = st.columns(2)
    with col1:
        num_col = st.selectbox("Numeric:", numeric_cols, key="anova_num")
    with col2:
        cat_col = st.selectbox("Category:", cat_cols, key="anova_cat")
    
    groups = [group[num_col].dropna().values for name, group in df.groupby(cat_col)]
    f_stat, p_value = f_oneway(*groups)
    
    results = {
        "F-statistic": f"{f_stat:.4f}",
        "p-value": f"{p_value:.6f}",
        "Significant": "Yes ✅" if p_value < 0.05 else "No"
    }
    
    st.dataframe(pd.DataFrame(list(results.items()), columns=['Metric', 'Value']), use_container_width=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    df.boxplot(column=num_col, by=cat_col, ax=ax)
    st.pyplot(fig)

def clustering_analysis(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        st.error("Need at least 2 numeric columns")
        return
    
    n_clusters = st.slider("Number of clusters:", 2, 10, 3, key="clusters")
    
    X = df[numeric_cols].fillna(df[numeric_cols].mean())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    results = {
        "Clusters": n_clusters,
        "Inertia": f"{kmeans.inertia_:.4f}",
        "Samples": len(df)
    }
    
    st.dataframe(pd.DataFrame(list(results.items()), columns=['Metric', 'Value']), use_container_width=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(X[numeric_cols[0]], X[numeric_cols[1]], c=clusters, cmap='viridis', s=50)
    ax.set_xlabel(numeric_cols[0])
    ax.set_ylabel(numeric_cols[1])
    plt.colorbar(scatter, ax=ax)
    st.pyplot(fig)

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    try:
        st.image("logo_1.png", width=250)
    except:
        st.title("📊 Data Insight")
    st.divider()
    
    if not st.session_state.user_id:
        login_type = st.radio("Login as:", ["Student", "Admin"])
        
        if login_type == "Student":
            st.write("Demo: student@example.com / password")
            email = st.text_input("Email:")
            pwd = st.text_input("Password:", type="password")
            if st.button("Sign In", use_container_width=True):
                if email == "student@example.com" and pwd == "password":
                    st.session_state.user_id = f"student_{email}"
                    st.session_state.user_name = "Student"
                    st.rerun()
                else:
                    st.error("Invalid")
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
        if st.button("📚 Homework", use_container_width=True):
            st.session_state.current_page = "homework"
            st.rerun()
        if st.session_state.user_id == "admin":
            if st.button("⚙️ Admin", use_container_width=True):
                st.session_state.current_page = "admin"
                st.rerun()
        
        st.divider()
        if st.button("🚪 Sign Out", use_container_width=True):
            st.session_state.user_id = None
            st.rerun()

# ============================================================================
# LANDING PAGE
# ============================================================================

if not st.session_state.user_id and st.session_state.current_page == "home":
    st.title("📊 Data Insight Studio")
    st.subheader("Professional Statistics & ML Helper")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.write("Master statistics with AI guidance. Upload data, ask questions, learn!")
    with col2:
        st.warning("⚖️ Do NOT use on exams!")
    
    st.divider()
    st.write("## Tools Available")
    cols = st.columns(2)
    tools = [
        ("📊 Descriptive Stats", "Mean, median, visualizations"),
        ("✨ Normality Testing", "Test & transform data"),
        ("📉 Regression", "Linear regression"),
        ("📋 Correlation", "Heatmaps"),
        ("📌 ANOVA", "Group comparisons"),
        ("📊 Clustering", "K-Means patterns"),
    ]
    for i, (title, desc) in enumerate(tools):
        with cols[i % 2]:
            st.write(f"### {title}\n{desc}")
    
    st.success("👉 Sign in to start!")

# ============================================================================
# HOMEWORK HELP
# ============================================================================

elif st.session_state.user_id and st.session_state.current_page == "homework":
    st.header("📚 Homework Help")
    
    # Input section
    st.write("### 1️⃣ Choose Analysis Type:")
    analysis_type = st.selectbox(
        "Select:",
        ["Descriptive Statistics", "Normality Testing", "Regression Analysis", 
         "Correlation Analysis", "ANOVA", "Clustering Analysis"],
        key="analysis_select"
    )
    
    st.divider()
    
    st.write("### 2️⃣ Upload File OR Type Question:")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Upload CSV/Excel/Image:", type=["csv", "xlsx", "xls", "jpg", "jpeg", "png"])
    
    with col2:
        question = st.text_area("Or type question:", placeholder="Type your statistics question here...", height=120)
    
    st.divider()
    
    st.write("### 3️⃣ Get Results:")
    
    if st.button("🚀 RUN ANALYSIS", use_container_width=True, key="run_button"):
        if uploaded_file:
            if uploaded_file.name.endswith(('jpg', 'jpeg', 'png')):
                st.image(uploaded_file, use_container_width=True)
                st.info("Image uploaded")
            else:
                df = read_file(uploaded_file)
                if df is not None:
                    st.write("**Data Preview:**")
                    st.dataframe(df.head(), use_container_width=True)
                    
                    st.divider()
                    st.write(f"### {analysis_type}")
                    
                    try:
                        if analysis_type == "Descriptive Statistics":
                            descriptive_stats(df)
                        elif analysis_type == "Normality Testing":
                            normality_test(df)
                        elif analysis_type == "Regression Analysis":
                            regression_analysis(df)
                        elif analysis_type == "Correlation Analysis":
                            correlation_analysis(df)
                        elif analysis_type == "ANOVA":
                            anova_analysis(df)
                        elif analysis_type == "Clustering Analysis":
                            clustering_analysis(df)
                        st.success("✅ Analysis complete!")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        elif question:
            api_key = get_api_key()
            if not api_key:
                st.error("❌ Admin must set API key first!")
            else:
                with st.spinner("🤖 Thinking..."):
                    answer = solve_with_ai(question, analysis_type, api_key)
                st.divider()
                st.markdown(answer)
                st.success("✅ Answer provided!")
        
        else:
            st.error("❌ Please upload file or type question!")

# ============================================================================
# HOME (LOGGED IN)
# ============================================================================

elif st.session_state.user_id and st.session_state.current_page == "home":
    st.title("📊 Data Insight Studio")
    st.write("Click **Homework** in sidebar to get started!")

# ============================================================================
# ADMIN PANEL
# ============================================================================

elif st.session_state.user_id == "admin" and st.session_state.current_page == "admin":
    st.title("⚙️ Admin Panel")
    st.divider()
    
    st.write("### 🔑 Configure API Key")
    
    current_key = load_api_key()
    if current_key:
        st.success(f"✅ Active: {current_key[:20]}...")
    else:
        st.warning("⚠️ No API key configured")
    
    st.divider()
    
    api_key_input = st.text_input(
        "API Key (sk-ant-...):",
        type="password",
        value=current_key or "",
        help="Get from https://console.anthropic.com/account/keys"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Save", use_container_width=True):
            if api_key_input:
                if save_api_key(api_key_input):
                    st.session_state.api_key = api_key_input
                    st.success("✅ Saved!")
                    st.rerun()
            else:
                st.error("Enter key first")
    
    with col2:
        if st.button("🗑️ Delete", use_container_width=True):
            try:
                if os.path.exists(CONFIG_FILE):
                    os.remove(CONFIG_FILE)
                st.session_state.api_key = None
                st.success("✅ Deleted!")
                st.rerun()
            except:
                st.error("Error")
    
    st.divider()
    st.write("### Instructions:")
    st.markdown("""
    1. Get key: https://console.anthropic.com/account/keys
    2. Paste above
    3. Click Save
    4. Done! ✅
    """)
