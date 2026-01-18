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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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
    """Save API key to file"""
    try:
        with open(CONFIG_FILE, "w") as f:
            f.write(key.strip())
        return True
    except Exception as e:
        st.error(f"Error saving: {str(e)}")
        return False

def load_api_key():
    """Load API key from environment or file"""
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
    """Get API key from session state or file"""
    if st.session_state.api_key:
        return st.session_state.api_key
    file_key = load_api_key()
    if file_key:
        st.session_state.api_key = file_key
        return file_key
    return None

def test_api_key(api_key):
    """Test if API key is valid"""
    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
            json={
                "model": "claude-opus-4-1-20250805",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "Say 'API works!'"}]
            },
            timeout=10
        )
        return response.status_code == 200
    except:
        return False

# ============================================================================
# SESSION STATE
# ============================================================================

if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "user_name" not in st.session_state:
    st.session_state.user_name = None
if "api_key" not in st.session_state:
    st.session_state.api_key = load_api_key()
if "current_page" not in st.session_state:
    st.session_state.current_page = "home"
if "api_calls" not in st.session_state:
    st.session_state.api_calls = 0

# ============================================================================
# AI PROBLEM SOLVER
# ============================================================================

def solve_with_ai(problem_text, category, api_key):
    """Use Claude API to solve the problem"""
    try:
        if not api_key:
            st.error("❌ API key not configured!")
            return None
        
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
                "system": f"You are an expert statistics tutor. Help students understand concepts. Category: {category}",
                "messages": [{"role": "user", "content": problem_text}]
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            st.session_state.api_calls += 1
            return result['content'][0]['text']
        else:
            st.error(f"API Error {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

# ============================================================================
# HELPERS
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
    return {
        "Mean": df[column].mean(),
        "Median": df[column].median(),
        "Std Dev": df[column].std(),
        "Min": df[column].min(),
        "Max": df[column].max(),
        "Q1": df[column].quantile(0.25),
        "Q3": df[column].quantile(0.75),
        "N": len(df)
    }

def create_visualizations(df, column):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0, 0].hist(df[column].dropna(), bins=30, edgecolor='black', alpha=0.7)
    axes[0, 0].set_title('Histogram')
    axes[0, 1].boxplot(df[column].dropna())
    axes[0, 1].set_title('Box Plot')
    stats.probplot(df[column].dropna(), dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot')
    df[column].plot(kind='density', ax=axes[1, 1])
    axes[1, 1].set_title('Density')
    plt.tight_layout()
    return fig

def test_normality(data):
    data_clean = data.dropna()
    shapiro_stat, shapiro_p = shapiro(data_clean)
    return {
        "Shapiro-Wilk p-value": f"{shapiro_p:.6f}",
        "Skewness": f"{stats.skew(data_clean):.4f}",
        "Is Normal?": "Yes ✅" if shapiro_p > 0.05 else "No ❌"
    }

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

if st.session_state.current_page == "home" and not st.session_state.user_id:
    st.title("📊 Data Insight Studio")
    st.subheader("Professional Statistics & Machine Learning Helper")
    st.divider()
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.write("""
        ### Welcome! 🎓
        Master statistics with AI-guided learning.
        Upload data, ask questions, get instant help!
        
        **Pricing:** $14.99/term
        """)
    with col2:
        st.warning("⚖️ Do NOT use on exams!")
    
    st.divider()
    st.write("## 🚀 Tools")
    cols = st.columns(2)
    tools = [
        ("📊 Descriptive Stats", "Mean, median, SD, plots"),
        ("✨ Normality Testing", "Test & transform data"),
        ("📉 Regression", "Linear regression analysis"),
        ("📋 Correlation", "Heatmaps & relationships"),
        ("📌 ANOVA", "Group comparisons"),
        ("📊 Clustering", "K-Means patterns"),
    ]
    for idx, (title, desc) in enumerate(tools):
        with cols[idx % 2]:
            st.write(f"### {title}\n{desc}")
    
    st.divider()
    st.success("👉 Sign in to start!")

# ============================================================================
# HOMEWORK HELP PAGE
# ============================================================================

elif st.session_state.current_page == "homework" and st.session_state.user_id:
    st.header("📚 Homework Help")
    st.write("Upload file or type question → Select analysis → Get results!")
    st.divider()
    
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
        if st.button("📊 Clustering", use_container_width=True):
            st.session_state.selected_category = "Clustering"
    
    st.divider()
    
    col1, col2 = st.columns([1, 1])
    with col1:
        uploaded_file = st.file_uploader("Upload:", type=["csv", "xlsx", "xls", "jpg", "jpeg", "png"])
    with col2:
        problem_text = st.text_area("Or type question:", height=100, placeholder="Type your question...")
    
    st.divider()
    
    if "selected_category" in st.session_state:
        st.write(f"### {st.session_state.selected_category}")
        
        df = None
        if uploaded_file:
            if uploaded_file.name.endswith(('jpg', 'jpeg', 'png')):
                st.image(uploaded_file, use_container_width=True)
            else:
                df = read_excel_csv(uploaded_file)
                if df is not None:
                    st.dataframe(df.head(), use_container_width=True)
        
        if st.button("🚀 RUN ANALYSIS", use_container_width=True):
            if df is not None:
                try:
                    if st.session_state.selected_category == "Descriptive Statistics":
                        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                        if numeric_cols:
                            col = numeric_cols[0]
                            st.dataframe(pd.DataFrame(list(calculate_descriptive_stats(df, col).items()), columns=['Stat', 'Value']), use_container_width=True)
                            st.pyplot(create_visualizations(df, col))
                except Exception as e:
                    st.error(f"Error: {str(e)}")
            elif problem_text:
                api_key = get_api_key()
                if not api_key:
                    st.error("❌ Admin must configure API key!")
                else:
                    with st.spinner("🤖 Thinking..."):
                        solution = solve_with_ai(problem_text, st.session_state.selected_category, api_key)
                    if solution:
                        st.markdown(solution)
                        st.success("✅ Done!")
            else:
                st.warning("Upload file or type question!")
    else:
        st.info("👆 Select analysis type!")

# ============================================================================
# HOME PAGE (LOGGED IN)
# ============================================================================

elif st.session_state.current_page == "home" and st.session_state.user_id:
    st.title("📊 Data Insight Studio")
    st.write("Welcome! Click **Homework** in sidebar to get started.")

# ============================================================================
# ADMIN PANEL
# ============================================================================

elif st.session_state.current_page == "admin" and st.session_state.user_id == "admin":
    st.title("⚙️ Admin Panel")
    st.divider()
    
    st.write("### 🔑 API Key Configuration")
    
    # Get current key
    current_key = load_api_key()
    
    # Show status
    if current_key:
        st.success(f"✅ API Key Active: {current_key[:20]}...")
    else:
        st.warning("⚠️ No API key configured")
    
    st.divider()
    
    # Input for new key
    st.write("**Enter or Update API Key:**")
    new_api_key = st.text_input(
        "API Key (sk-ant-...)",
        type="password",
        value=current_key or "",
        placeholder="sk-ant-...",
        help="Get from https://console.anthropic.com/account/keys"
    )
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("💾 Save API Key", use_container_width=True):
            if new_api_key:
                if save_api_key(new_api_key):
                    st.session_state.api_key = new_api_key
                    st.success("✅ API Key Saved Successfully!")
                else:
                    st.error("❌ Failed to save API key")
            else:
                st.error("❌ Please enter API key")
    
    with col2:
        if st.button("🧪 Test API Key", use_container_width=True):
            if new_api_key:
                with st.spinner("Testing..."):
                    if test_api_key(new_api_key):
                        st.success("✅ API Key Works!")
                    else:
                        st.error("❌ API Key Invalid!")
            else:
                st.error("❌ Please enter API key first")
    
    st.divider()
    
    # Clear key option
    if st.button("🗑️ Delete API Key", use_container_width=True):
        try:
            if os.path.exists(CONFIG_FILE):
                os.remove(CONFIG_FILE)
            st.session_state.api_key = None
            st.success("✅ API Key Deleted")
        except:
            st.error("❌ Error deleting key")
    
    st.divider()
    
    # Metrics
    st.write("### 📊 Metrics")
    col1, col2 = st.columns(2)
    col1.metric("API Calls", st.session_state.api_calls)
    col2.metric("Students", "Ready!")
    
    st.divider()
    
    # Instructions
    st.write("### 📝 Setup Instructions")
    st.markdown("""
    1. Get API key from: https://console.anthropic.com/account/keys
    2. Paste key above
    3. Click "Save API Key"
    4. Click "Test API Key" to verify
    5. Done! ✅
    
    Students can now use the app to ask questions!
    """)

# ============================================================================
# FALLBACK
# ============================================================================

else:
    if st.session_state.user_id:
        st.title("📊 Data Insight Studio")
        st.write("Select menu option in sidebar")
    else:
        st.title("Please Sign In")
        st.write("Use sidebar to login")
