import streamlit as st
import requests
import os
import pandas as pdimport streamlit as st
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
if "analysis_type" not in st.session_state:
    st.session_state.analysis_type = "Descriptive Statistics"
if "uploaded_df" not in st.session_state:
    st.session_state.uploaded_df = None
if "question_text" not in st.session_state:
    st.session_state.question_text = ""
if "show_results" not in st.session_state:
    st.session_state.show_results = False
if "transformed_df" not in st.session_state:
    st.session_state.transformed_df = None

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
            return "❌ ERROR: Admin must configure API key first!"
        
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
                "system": f"You are expert statistics tutor. Category: {category}",
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
# TRANSFORMATION & SAMPLING FUNCTIONS
# ============================================================================

def get_data_characteristics(data):
    """Analyze data to recommend best transformation"""
    data_clean = data.dropna()
    skewness = stats.skew(data_clean)
    kurtosis = stats.kurtosis(data_clean)
    
    characteristics = {
        "mean": data_clean.mean(),
        "std": data_clean.std(),
        "min": data_clean.min(),
        "max": data_clean.max(),
        "skewness": skewness,
        "kurtosis": kurtosis,
        "has_negative": (data_clean < 0).any(),
        "has_zero": (data_clean == 0).any()
    }
    
    return characteristics

def recommend_transformation(characteristics):
    """Recommend best transformation based on data"""
    skew = abs(characteristics["skewness"])
    
    recommendations = []
    
    if skew > 2:
        if not characteristics["has_negative"] and not characteristics["has_zero"]:
            recommendations.append(("Log", "🟢 BEST: Excellent for right-skewed data (positive values only)"))
            recommendations.append(("Box-Cox", "🟡 GOOD: Optimal automatic transformation"))
            recommendations.append(("Yeo-Johnson", "🟡 GOOD: Works with any values"))
        else:
            recommendations.append(("Yeo-Johnson", "🟢 BEST: Works with negative/zero values"))
            recommendations.append(("Box-Cox", "🟡 GOOD: Only with positive values"))
    elif skew > 1:
        if not characteristics["has_negative"] and not characteristics["has_zero"]:
            recommendations.append(("Square Root", "🟢 BEST: Less aggressive than log"))
            recommendations.append(("Log", "🟡 GOOD: Also works well"))
        else:
            recommendations.append(("Yeo-Johnson", "🟢 BEST: Flexible transformation"))
    else:
        recommendations.append(("Z-Score", "🟢 BEST: Data already near normal"))
        recommendations.append(("Yeo-Johnson", "🟡 GOOD: Fine-tune if needed"))
    
    return recommendations

def apply_transformation(data, method):
    """Apply selected transformation"""
    data_clean = data.dropna()
    
    if method == "Log":
        if (data_clean > 0).all():
            transformed = np.log(data_clean)
            explanation = "**Log Transformation**: Compresses right-skewed data. Works only with positive values. Great for multiplicative relationships."
            return transformed, explanation, True
        else:
            return None, "❌ Log requires all positive values!", False
    
    elif method == "Square Root":
        if (data_clean >= 0).all():
            transformed = np.sqrt(data_clean)
            explanation = "**Square Root**: Less aggressive than log. Good for moderate skewness and count data."
            return transformed, explanation, True
        else:
            return None, "❌ Square Root requires non-negative values!", False
    
    elif method == "Box-Cox":
        if (data_clean > 0).all():
            try:
                transformed, lambda_param = boxcox(data_clean)
                explanation = f"**Box-Cox (λ={lambda_param:.4f})**: Automatically finds optimal transformation. λ=0 means log, λ=0.5 means sqrt."
                return transformed, explanation, True
            except:
                return None, "❌ Box-Cox failed!", False
        else:
            return None, "❌ Box-Cox requires positive values!", False
    
    elif method == "Yeo-Johnson":
        try:
            transformed, lambda_param = yeojohnson(data_clean)
            explanation = f"**Yeo-Johnson (λ={lambda_param:.4f})**: Works with ANY values including negative/zero. More flexible than Box-Cox."
            return transformed, explanation, True
        except:
            return None, "❌ Yeo-Johnson failed!", False
    
    elif method == "Z-Score":
        mean = data_clean.mean()
        std = data_clean.std()
        transformed = (data_clean - mean) / std
        explanation = "**Z-Score Standardization**: Centers data (mean=0) and scales (std=1). Preserves distribution shape."
        return transformed, explanation, True
    
    elif method == "Min-Max":
        min_val = data_clean.min()
        max_val = data_clean.max()
        transformed = (data_clean - min_val) / (max_val - min_val)
        explanation = "**Min-Max Normalization**: Scales data to [0,1] range. Preserves relationships."
        return transformed, explanation, True
    
    return None, "", False

def test_normality(data):
    """Test if data is normally distributed"""
    data_clean = data.dropna()
    shapiro_stat, shapiro_p = shapiro(data_clean)
    anderson_result = anderson(data_clean)
    
    return {
        "Shapiro-Wilk p-value": f"{shapiro_p:.6f}",
        "Anderson-Darling": f"{anderson_result.statistic:.6f}",
        "Skewness": f"{stats.skew(data_clean):.4f}",
        "Kurtosis": f"{stats.kurtosis(data_clean):.4f}",
        "Normal?": "✅ YES" if shapiro_p > 0.05 else "❌ NO"
    }

def bootstrap_sample(df, column, n_samples=None):
    """Create bootstrap samples to increase sample size"""
    if n_samples is None:
        n_samples = len(df)
    
    data = df[column].dropna()
    bootstrapped = np.random.choice(data, size=n_samples, replace=True)
    return pd.Series(bootstrapped)

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
        st.error(f"Error: {str(e)}")
        return None

def descriptive_stats(df):
    """Descriptive Statistics Analysis"""
    st.write("### 📊 Descriptive Statistics Analysis")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        st.error("No numeric columns found")
        return
    
    col = st.selectbox("Select column:", numeric_cols, key="desc_col_select")
    col_data = df[col].dropna()
    
    stats_dict = {
        "Mean": f"{col_data.mean():.4f}",
        "Median": f"{col_data.median():.4f}",
        "Std Dev": f"{col_data.std():.4f}",
        "Min": f"{col_data.min():.4f}",
        "Q1": f"{col_data.quantile(0.25):.4f}",
        "Q3": f"{col_data.quantile(0.75):.4f}",
        "Max": f"{col_data.max():.4f}",
        "Skewness": f"{stats.skew(col_data):.4f}",
        "Kurtosis": f"{stats.kurtosis(col_data):.4f}",
        "Count": len(col_data)
    }
    
    st.write("**Statistics:**")
    st.dataframe(pd.DataFrame(list(stats_dict.items()), columns=['Statistic', 'Value']), use_container_width=True)
    
    st.write("**Visualizations:**")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].hist(col_data, bins=30, edgecolor='black', alpha=0.7)
    axes[0, 0].set_title(f'Histogram of {col}')
    axes[0, 1].boxplot(col_data)
    axes[0, 1].set_title(f'Box Plot of {col}')
    stats.probplot(col_data, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot')
    col_data.plot(kind='density', ax=axes[1, 1])
    axes[1, 1].set_title(f'Density Plot')
    
    plt.tight_layout()
    st.pyplot(fig)
    st.success("✅ Analysis complete!")

def normality_and_transform(df):
    """Normality Testing with Smart Transformations"""
    st.write("### ✨ Normality Testing & Transformations")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        st.error("No numeric columns found")
        return
    
    col = st.selectbox("Select column:", numeric_cols, key="norm_col_select")
    data = df[col].dropna()
    
    # Test original data
    st.write("#### 1️⃣ Original Data Analysis")
    original_results = test_normality(data)
    st.dataframe(pd.DataFrame(list(original_results.items()), columns=['Test', 'Result']), use_container_width=True)
    
    # Get characteristics and recommendations
    characteristics = get_data_characteristics(data)
    recommendations = recommend_transformation(characteristics)
    
    st.write("#### 2️⃣ Transformation Recommendations")
    for trans_name, explanation in recommendations:
        st.write(f"- {explanation}")
    
    # Visualization of original
    st.write("#### 3️⃣ Original Data Visualization")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0, 0].hist(data, bins=30, edgecolor='black', alpha=0.7)
    axes[0, 0].set_title('Histogram (Original)')
    axes[0, 1].boxplot(data)
    axes[0, 1].set_title('Box Plot (Original)')
    stats.probplot(data, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot (Original)')
    data.plot(kind='density', ax=axes[1, 1])
    axes[1, 1].set_title('Density (Original)')
    plt.tight_layout()
    st.pyplot(fig)
    
    # Select transformation
    st.write("#### 4️⃣ Apply Transformation")
    trans_method = st.selectbox(
        "Select transformation method:",
        ["Log", "Square Root", "Box-Cox", "Yeo-Johnson", "Z-Score", "Min-Max"],
        index=0,
        key="trans_select"
    )
    
    transformed_data, explanation, success = apply_transformation(data, trans_method)
    
    if success:
        st.info(explanation)
        
        # Test transformed data
        st.write("#### 5️⃣ Transformed Data Analysis")
        trans_results = test_normality(transformed_data)
        st.dataframe(pd.DataFrame(list(trans_results.items()), columns=['Test', 'Result']), use_container_width=True)
        
        # Improvement
        original_p = float(original_results["Shapiro-Wilk p-value"])
        trans_p = float(trans_results["Shapiro-Wilk p-value"])
        improvement = ((trans_p - original_p) / abs(original_p)) * 100 if original_p != 0 else 0
        
        if trans_p > 0.05:
            st.success(f"✅ Transformation SUCCESSFUL! (improvement: {improvement:+.1f}%)")
        else:
            st.info(f"Data improved but still not perfectly normal (p={trans_p:.4f})")
        
        # Before/After visualization
        st.write("#### 6️⃣ Before & After Comparison")
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        axes[0, 0].hist(data, bins=30, edgecolor='black')
        axes[0, 0].set_title('Original: Histogram')
        stats.probplot(data, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Original: Q-Q Plot')
        data.plot(kind='density', ax=axes[0, 2])
        axes[0, 2].set_title('Original: Density')
        
        axes[1, 0].hist(transformed_data, bins=30, edgecolor='black')
        axes[1, 0].set_title(f'{trans_method}: Histogram')
        stats.probplot(transformed_data, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title(f'{trans_method}: Q-Q Plot')
        transformed_data.plot(kind='density', ax=axes[1, 2])
        axes[1, 2].set_title(f'{trans_method}: Density')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Sampling & Dataset Update
        st.write("#### 7️⃣ Sampling & Dataset Update")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            use_sampling = st.checkbox("Apply Bootstrap Sampling?", value=False)
        with col2:
            sample_multiplier = st.slider("Sample Size Multiplier:", 1, 5, 2, help="1x = original size, 2x = double, etc")
        
        if use_sampling:
            sample_size = len(data) * sample_multiplier
            st.write(f"Original size: {len(data)} → New size: {int(sample_size)}")
            
            sampled = bootstrap_sample(df, col, n_samples=int(sample_size))
            sampled_trans, _, _ = apply_transformation(sampled, trans_method)
            
            st.write("**Sampled + Transformed Data Tests:**")
            sampled_results = test_normality(sampled_trans)
            st.dataframe(pd.DataFrame(list(sampled_results.items()), columns=['Test', 'Result']), use_container_width=True)
            
            # Create updated dataset
            updated_df = df.copy()
            
            # Add sampled column
            if sample_size > len(data):
                # Expand dataframe with new sampled data
                new_rows = int(sample_size - len(data))
                new_data = bootstrap_sample(df, col, n_samples=new_rows)
                new_rows_df = pd.DataFrame({col: new_data.values[:new_rows]})
                updated_df = pd.concat([updated_df, new_rows_df], ignore_index=True)
            
            # Replace column with transformed values
            updated_df[f"{col}_transformed"] = updated_df[col].fillna(df[col].mean())
            scaler = StandardScaler()
            for numeric_col in updated_df.select_dtypes(include=[np.number]).columns:
                if numeric_col != f"{col}_transformed":
                    updated_df[numeric_col] = scaler.fit_transform(updated_df[[numeric_col]])
            
            st.session_state.transformed_df = updated_df
            
            st.write("**Updated Dataset Preview:**")
            st.dataframe(updated_df.head(10), use_container_width=True)
            st.metric("New Dataset Size", len(updated_df))
            
            # Download button
            csv = updated_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Updated Dataset",
                data=csv,
                file_name=f"transformed_sampled_{col}.csv",
                mime="text/csv"
            )
            
            st.success("✅ Dataset updated! Download and use for further analysis.")
        
        st.success("✅ Transformation analysis complete!")
    else:
        st.error(explanation)

def regression_analysis(df):
    """Regression Analysis"""
    st.write("### 📉 Regression Analysis")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        st.error("Need at least 2 numeric columns")
        return
    
    col1, col2 = st.columns(2)
    with col1:
        x_col = st.selectbox("X (Independent):", numeric_cols, key="reg_x_select")
    with col2:
        y_col = st.selectbox("Y (Dependent):", numeric_cols, key="reg_y_select", index=min(1, len(numeric_cols)-1))
    
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
        "R² Value": f"{r_sq:.4f}",
        "Slope": f"{slope:.4f}",
        "Intercept": f"{intercept:.4f}"
    }
    
    st.dataframe(pd.DataFrame(list(results.items()), columns=['Metric', 'Value']), use_container_width=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X, y, alpha=0.6, label='Data')
    X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_line = model.predict(X_line)
    ax.plot(X_line, y_line, 'r-', linewidth=2, label=f'Y = {intercept:.2f} + {slope:.2f}*X')
    ax.set_xlabel(x_col, fontsize=12)
    ax.set_ylabel(y_col, fontsize=12)
    ax.set_title('Linear Regression', fontsize=14)
    ax.legend()
    st.pyplot(fig)
    st.success("✅ Analysis complete!")

def correlation_analysis(df):
    """Correlation Analysis"""
    st.write("### 📋 Correlation Analysis")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        st.error("Need at least 2 numeric columns")
        return
    
    corr = df[numeric_cols].corr()
    st.dataframe(corr, use_container_width=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
    st.pyplot(fig)
    st.success("✅ Analysis complete!")

def anova_analysis(df):
    """ANOVA Analysis"""
    st.write("### 📌 ANOVA Analysis")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if not numeric_cols or not cat_cols:
        st.error("Need numeric and categorical columns")
        return
    
    col1, col2 = st.columns(2)
    with col1:
        num_col = st.selectbox("Numeric Variable:", numeric_cols, key="anova_num_select")
    with col2:
        cat_col = st.selectbox("Categorical Variable:", cat_cols, key="anova_cat_select")
    
    groups = [group[num_col].dropna().values for name, group in df.groupby(cat_col)]
    
    if len(groups) < 2:
        st.error("Need at least 2 groups")
        return
    
    f_stat, p_value = f_oneway(*groups)
    
    results = {
        "F-statistic": f"{f_stat:.4f}",
        "p-value": f"{p_value:.6f}",
        "Significant": "✅ YES" if p_value < 0.05 else "❌ NO"
    }
    
    st.dataframe(pd.DataFrame(list(results.items()), columns=['Metric', 'Value']), use_container_width=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    df.boxplot(column=num_col, by=cat_col, ax=ax)
    plt.suptitle('')
    st.pyplot(fig)
    st.success("✅ Analysis complete!")

def clustering_analysis(df):
    """Clustering Analysis"""
    st.write("### 📊 Clustering Analysis")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        st.error("Need at least 2 numeric columns")
        return
    
    n_clusters = st.slider("Number of clusters:", 2, 10, 3, key="cluster_slider")
    
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
    
    st.dataframe(pd.DataFrame(list(results.items()), columns=['Metric', 'Value']), use_container_width=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(X[numeric_cols[0]], X[numeric_cols[1]], c=clusters, cmap='viridis', s=50, alpha=0.6)
    ax.set_xlabel(numeric_cols[0], fontsize=12)
    ax.set_ylabel(numeric_cols[1], fontsize=12)
    ax.set_title(f'K-Means Clustering (k={n_clusters})', fontsize=14)
    plt.colorbar(scatter, ax=ax, label='Cluster')
    st.pyplot(fig)
    st.success("✅ Analysis complete!")

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
        login_type = st.radio("Login as:", ["Student", "Admin"], key="login_radio")
        
        if login_type == "Student":
            st.write("Demo: student@example.com / password")
            email = st.text_input("Email:", key="student_email_input")
            pwd = st.text_input("Password:", type="password", key="student_pwd_input")
            if st.button("Sign In", use_container_width=True, key="student_signin"):
                if email == "student@example.com" and pwd == "password":
                    st.session_state.user_id = f"student_{email}"
                    st.session_state.user_name = "Student"
                    st.rerun()
                else:
                    st.error("Invalid")
        else:
            pwd = st.text_input("Admin Password:", type="password", key="admin_pwd_input")
            if st.button("Sign In", use_container_width=True, key="admin_signin"):
                if pwd == "admin123":
                    st.session_state.user_id = "admin"
                    st.session_state.user_name = "Admin"
                    st.rerun()
                else:
                    st.error("Invalid")
    else:
        st.write(f"### {st.session_state.user_name}! 👋")
        st.divider()
        
        if st.button("📊 Home", use_container_width=True, key="home_btn"):
            st.session_state.current_page = "home"
            st.rerun()
        if st.button("📚 Homework", use_container_width=True, key="homework_btn"):
            st.session_state.current_page = "homework"
            st.rerun()
        if st.session_state.user_id == "admin":
            if st.button("⚙️ Admin", use_container_width=True, key="admin_btn"):
                st.session_state.current_page = "admin"
                st.rerun()
        
        st.divider()
        if st.button("🚪 Sign Out", use_container_width=True, key="signout_btn"):
            st.session_state.user_id = None
            st.session_state.current_page = "home"
            st.rerun()

# ============================================================================
# LANDING PAGE
# ============================================================================

if not st.session_state.user_id and st.session_state.current_page == "home":
    st.title("📊 Data Insight Studio")
    st.subheader("Professional Statistics & ML Helper")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.write("Master statistics with AI guidance!")
    with col2:
        st.warning("⚖️ Do NOT use on exams!")
    
    st.divider()
    st.write("## Tools")
    cols = st.columns(2)
    tools = [
        ("📊 Descriptive Stats", "Mean, median, plots"),
        ("✨ Normality & Transform", "Test, transform, sample data"),
        ("📉 Regression", "Linear fit"),
        ("📋 Correlation", "Heatmaps"),
        ("📌 ANOVA", "Group compare"),
        ("📊 Clustering", "K-Means"),
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
    
    st.write("### Step 1️⃣: Choose Analysis Type")
    st.session_state.analysis_type = st.selectbox(
        "Select Analysis:",
        ["Descriptive Statistics", "Normality Testing & Transformations", "Regression Analysis", 
         "Correlation Analysis", "ANOVA", "Clustering Analysis"],
        index=["Descriptive Statistics", "Normality Testing & Transformations", "Regression Analysis", 
               "Correlation Analysis", "ANOVA", "Clustering Analysis"].index(st.session_state.analysis_type),
        key="analysis_dropdown"
    )
    
    st.divider()
    
    st.write("### Step 2️⃣: Upload File OR Type Question")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload CSV/Excel/Image:", 
            type=["csv", "xlsx", "xls", "jpg", "jpeg", "png"],
            key="file_upload"
        )
    
    with col2:
        st.session_state.question_text = st.text_area(
            "Or type question:", 
            value=st.session_state.question_text,
            placeholder="Type your statistics question...", 
            height=120,
            key="question_input"
        )
    
    st.divider()
    
    st.write("### Step 3️⃣: Run Analysis")
    
    if st.button("🚀 RUN ANALYSIS", use_container_width=True, key="run_button"):
        if uploaded_file:
            if uploaded_file.name.endswith(('jpg', 'jpeg', 'png')):
                st.image(uploaded_file, use_container_width=True)
                st.info("Image uploaded - describe what you need help with")
            else:
                st.session_state.uploaded_df = read_file(uploaded_file)
                
                if st.session_state.uploaded_df is not None:
                    st.write("**Data Preview:**")
                    st.dataframe(st.session_state.uploaded_df.head(), use_container_width=True)
                    st.session_state.show_results = True
        
        elif st.session_state.question_text:
            api_key = get_api_key()
            if not api_key:
                st.error("❌ Admin must configure API key!")
            else:
                with st.spinner("🤖 Thinking..."):
                    answer = solve_with_ai(st.session_state.question_text, st.session_state.analysis_type, api_key)
                st.divider()
                st.markdown(answer)
                st.success("✅ Done!")
        else:
            st.error("❌ Upload file or type question!")
    
    # Show results
    if st.session_state.show_results and st.session_state.uploaded_df is not None:
        st.divider()
        st.write(f"### Analysis: {st.session_state.analysis_type}")
        
        try:
            if st.session_state.analysis_type == "Descriptive Statistics":
                descriptive_stats(st.session_state.uploaded_df)
            elif st.session_state.analysis_type == "Normality Testing & Transformations":
                normality_and_transform(st.session_state.uploaded_df)
            elif st.session_state.analysis_type == "Regression Analysis":
                regression_analysis(st.session_state.uploaded_df)
            elif st.session_state.analysis_type == "Correlation Analysis":
                correlation_analysis(st.session_state.uploaded_df)
            elif st.session_state.analysis_type == "ANOVA":
                anova_analysis(st.session_state.uploaded_df)
            elif st.session_state.analysis_type == "Clustering Analysis":
                clustering_analysis(st.session_state.uploaded_df)
        except Exception as e:
            st.error(f"Error: {str(e)}")

# ============================================================================
# HOME (LOGGED IN)
# ============================================================================

elif st.session_state.user_id and st.session_state.current_page == "home":
    st.title("📊 Data Insight Studio")
    st.write("Click **Homework** to get started!")

# ============================================================================
# ADMIN PANEL
# ============================================================================

elif st.session_state.user_id == "admin" and st.session_state.current_page == "admin":
    st.title("⚙️ Admin Panel")
    st.divider()
    
    st.write("### Configure API Key")
    
    current_key = load_api_key()
    if current_key:
        st.success(f"✅ Active: {current_key[:20]}...")
    else:
        st.warning("⚠️ No API key")
    
    st.divider()
    
    api_key_input = st.text_input(
        "API Key:",
        type="password",
        value=current_key or "",
        key="api_key_input"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Save Key", use_container_width=True, key="save_key_btn"):
            if api_key_input:
                if save_api_key(api_key_input):
                    st.session_state.api_key = api_key_input
                    st.success("✅ Saved!")
            else:
                st.error("Enter key")
    
    with col2:
        if st.button("🗑️ Delete Key", use_container_width=True, key="delete_key_btn"):
            try:
                if os.path.exists(CONFIG_FILE):
                    os.remove(CONFIG_FILE)
                st.session_state.api_key = None
                st.success("✅ Deleted!")
            except:
                st.error("Error")
    
    st.divider()
    st.write("**Get key:** https://console.anthropic.com/account/keys")
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
# PERSISTENT SESSION STATE
# ============================================================================

if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "user_name" not in st.session_state:
    st.session_state.user_name = None
if "api_key" not in st.session_state:
    st.session_state.api_key = None
if "current_page" not in st.session_state:
    st.session_state.current_page = "home"
if "analysis_type" not in st.session_state:
    st.session_state.analysis_type = "Descriptive Statistics"
if "uploaded_df" not in st.session_state:
    st.session_state.uploaded_df = None
if "question_text" not in st.session_state:
    st.session_state.question_text = ""
if "show_results" not in st.session_state:
    st.session_state.show_results = False

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
            return "❌ ERROR: Admin must configure API key first!"
        
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
                "system": f"You are expert statistics tutor. Help students understand. Category: {category}",
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
        st.error(f"Error: {str(e)}")
        return None

def descriptive_stats(df):
    """Descriptive Statistics Analysis"""
    st.write("### 📊 Descriptive Statistics Analysis")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        st.error("No numeric columns found")
        return
    
    col = st.selectbox("Select column:", numeric_cols, key="desc_col_select")
    
    col_data = df[col].dropna()
    
    stats_dict = {
        "Mean": f"{col_data.mean():.4f}",
        "Median": f"{col_data.median():.4f}",
        "Std Dev": f"{col_data.std():.4f}",
        "Min": f"{col_data.min():.4f}",
        "Q1": f"{col_data.quantile(0.25):.4f}",
        "Q3": f"{col_data.quantile(0.75):.4f}",
        "Max": f"{col_data.max():.4f}",
        "Skewness": f"{stats.skew(col_data):.4f}",
        "Kurtosis": f"{stats.kurtosis(col_data):.4f}",
        "Count": len(col_data)
    }
    
    st.write("**Statistics:**")
    st.dataframe(pd.DataFrame(list(stats_dict.items()), columns=['Statistic', 'Value']), use_container_width=True)
    
    st.write("**Visualizations:**")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].hist(col_data, bins=30, edgecolor='black', alpha=0.7)
    axes[0, 0].set_title(f'Histogram of {col}')
    axes[0, 0].set_ylabel('Frequency')
    
    axes[0, 1].boxplot(col_data)
    axes[0, 1].set_title(f'Box Plot of {col}')
    axes[0, 1].set_ylabel('Value')
    
    stats.probplot(col_data, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot')
    
    col_data.plot(kind='density', ax=axes[1, 1])
    axes[1, 1].set_title(f'Density Plot of {col}')
    
    plt.tight_layout()
    st.pyplot(fig)
    st.success("✅ Analysis complete!")

def normality_test(df):
    """Normality Testing Analysis"""
    st.write("### ✨ Normality Testing")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        st.error("No numeric columns found")
        return
    
    col = st.selectbox("Select column:", numeric_cols, key="norm_col_select")
    data = df[col].dropna()
    
    # Tests
    shapiro_stat, shapiro_p = shapiro(data)
    anderson_result = anderson(data)
    ks_stat, ks_p = kstest(data, 'norm', args=(data.mean(), data.std()))
    
    results = {
        "Shapiro-Wilk p-value": f"{shapiro_p:.6f}",
        "Anderson-Darling Stat": f"{anderson_result.statistic:.6f}",
        "K-S p-value": f"{ks_p:.6f}",
        "Skewness": f"{stats.skew(data):.4f}",
        "Kurtosis": f"{stats.kurtosis(data):.4f}",
        "Normal?": "✅ YES" if shapiro_p > 0.05 else "❌ NO"
    }
    
    st.write("**Normality Tests:**")
    st.dataframe(pd.DataFrame(list(results.items()), columns=['Test', 'Result']), use_container_width=True)
    
    st.write("**Visualization:**")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].hist(data, bins=30, edgecolor='black', alpha=0.7, density=True)
    mu, sigma = stats.norm.fit(data)
    x = np.linspace(data.min(), data.max(), 100)
    axes[0, 0].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2)
    axes[0, 0].set_title('Original Distribution')
    
    stats.probplot(data, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Q-Q Plot (Original)')
    
    axes[1, 0].boxplot(data)
    axes[1, 0].set_title('Box Plot')
    
    data.plot(kind='density', ax=axes[1, 1])
    axes[1, 1].set_title('Density Plot')
    
    plt.tight_layout()
    st.pyplot(fig)
    st.success("✅ Analysis complete!")

def regression_analysis(df):
    """Regression Analysis"""
    st.write("### 📉 Regression Analysis")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        st.error("Need at least 2 numeric columns")
        return
    
    col1, col2 = st.columns(2)
    with col1:
        x_col = st.selectbox("X (Independent):", numeric_cols, key="reg_x_select")
    with col2:
        y_col = st.selectbox("Y (Dependent):", numeric_cols, key="reg_y_select", index=min(1, len(numeric_cols)-1))
    
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
        "R² Value": f"{r_sq:.4f}",
        "Slope": f"{slope:.4f}",
        "Intercept": f"{intercept:.4f}"
    }
    
    st.write("**Results:**")
    st.dataframe(pd.DataFrame(list(results.items()), columns=['Metric', 'Value']), use_container_width=True)
    
    st.write("**Plot:**")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X, y, alpha=0.6, label='Data')
    X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_line = model.predict(X_line)
    ax.plot(X_line, y_line, 'r-', linewidth=2, label=f'Y = {intercept:.2f} + {slope:.2f}*X')
    ax.set_xlabel(x_col, fontsize=12)
    ax.set_ylabel(y_col, fontsize=12)
    ax.set_title('Linear Regression', fontsize=14)
    ax.legend()
    st.pyplot(fig)
    st.success("✅ Analysis complete!")

def correlation_analysis(df):
    """Correlation Analysis"""
    st.write("### 📋 Correlation Analysis")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        st.error("Need at least 2 numeric columns")
        return
    
    corr = df[numeric_cols].corr()
    
    st.write("**Correlation Matrix:**")
    st.dataframe(corr, use_container_width=True)
    
    st.write("**Heatmap:**")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
    st.pyplot(fig)
    st.success("✅ Analysis complete!")

def anova_analysis(df):
    """ANOVA Analysis"""
    st.write("### 📌 ANOVA Analysis")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if not numeric_cols or not cat_cols:
        st.error("Need numeric and categorical columns")
        return
    
    col1, col2 = st.columns(2)
    with col1:
        num_col = st.selectbox("Numeric Variable:", numeric_cols, key="anova_num_select")
    with col2:
        cat_col = st.selectbox("Categorical Variable:", cat_cols, key="anova_cat_select")
    
    groups = [group[num_col].dropna().values for name, group in df.groupby(cat_col)]
    
    if len(groups) < 2:
        st.error("Need at least 2 groups")
        return
    
    f_stat, p_value = f_oneway(*groups)
    
    results = {
        "F-statistic": f"{f_stat:.4f}",
        "p-value": f"{p_value:.6f}",
        "Significant (p<0.05)": "✅ YES" if p_value < 0.05 else "❌ NO"
    }
    
    st.write("**ANOVA Results:**")
    st.dataframe(pd.DataFrame(list(results.items()), columns=['Metric', 'Value']), use_container_width=True)
    
    st.write("**Box Plot by Group:**")
    fig, ax = plt.subplots(figsize=(10, 6))
    df.boxplot(column=num_col, by=cat_col, ax=ax)
    plt.suptitle('')
    st.pyplot(fig)
    st.success("✅ Analysis complete!")

def clustering_analysis(df):
    """Clustering Analysis"""
    st.write("### 📊 Clustering Analysis")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        st.error("Need at least 2 numeric columns")
        return
    
    n_clusters = st.slider("Number of clusters:", 2, 10, 3, key="cluster_slider")
    
    X = df[numeric_cols].fillna(df[numeric_cols].mean())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    results = {
        "Number of Clusters": n_clusters,
        "Inertia": f"{kmeans.inertia_:.4f}",
        "Total Samples": len(df)
    }
    
    st.write("**Clustering Results:**")
    st.dataframe(pd.DataFrame(list(results.items()), columns=['Metric', 'Value']), use_container_width=True)
    
    st.write("**Cluster Visualization:**")
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(X[numeric_cols[0]], X[numeric_cols[1]], c=clusters, cmap='viridis', s=50, alpha=0.6)
    ax.set_xlabel(numeric_cols[0], fontsize=12)
    ax.set_ylabel(numeric_cols[1], fontsize=12)
    ax.set_title(f'K-Means Clustering (k={n_clusters})', fontsize=14)
    plt.colorbar(scatter, ax=ax, label='Cluster')
    st.pyplot(fig)
    st.success("✅ Analysis complete!")

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
        login_type = st.radio("Login as:", ["Student", "Admin"], key="login_radio")
        
        if login_type == "Student":
            st.write("Demo: student@example.com / password")
            email = st.text_input("Email:", key="student_email_input")
            pwd = st.text_input("Password:", type="password", key="student_pwd_input")
            if st.button("Sign In", use_container_width=True, key="student_signin"):
                if email == "student@example.com" and pwd == "password":
                    st.session_state.user_id = f"student_{email}"
                    st.session_state.user_name = "Student"
                    st.rerun()
                else:
                    st.error("Invalid")
        else:
            pwd = st.text_input("Admin Password:", type="password", key="admin_pwd_input")
            if st.button("Sign In", use_container_width=True, key="admin_signin"):
                if pwd == "admin123":
                    st.session_state.user_id = "admin"
                    st.session_state.user_name = "Admin"
                    st.rerun()
                else:
                    st.error("Invalid")
    else:
        st.write(f"### {st.session_state.user_name}! 👋")
        st.divider()
        
        if st.button("📊 Home", use_container_width=True, key="home_btn"):
            st.session_state.current_page = "home"
            st.rerun()
        if st.button("📚 Homework", use_container_width=True, key="homework_btn"):
            st.session_state.current_page = "homework"
            st.rerun()
        if st.session_state.user_id == "admin":
            if st.button("⚙️ Admin", use_container_width=True, key="admin_btn"):
                st.session_state.current_page = "admin"
                st.rerun()
        
        st.divider()
        if st.button("🚪 Sign Out", use_container_width=True, key="signout_btn"):
            st.session_state.user_id = None
            st.session_state.current_page = "home"
            st.rerun()

# ============================================================================
# LANDING PAGE
# ============================================================================

if not st.session_state.user_id and st.session_state.current_page == "home":
    st.title("📊 Data Insight Studio")
    st.subheader("Professional Statistics & ML Helper")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.write("Master statistics with AI guidance!")
    with col2:
        st.warning("⚖️ Do NOT use on exams!")
    
    st.divider()
    st.write("## Tools")
    cols = st.columns(2)
    tools = [
        ("📊 Descriptive Stats", "Mean, median, plots"),
        ("✨ Normality Testing", "Test & visualize"),
        ("📉 Regression", "Linear fit"),
        ("📋 Correlation", "Heatmaps"),
        ("📌 ANOVA", "Group compare"),
        ("📊 Clustering", "K-Means"),
    ]
    for i, (title, desc) in enumerate(tools):
        with cols[i % 2]:
            st.write(f"### {title}\n{desc}")
    
    st.success("👉 Sign in to start!")

# ============================================================================
# HOMEWORK HELP - MAIN PAGE
# ============================================================================

elif st.session_state.user_id and st.session_state.current_page == "homework":
    st.header("📚 Homework Help")
    
    # Input section (stays visible)
    st.write("### Step 1️⃣: Choose Analysis Type")
    st.session_state.analysis_type = st.selectbox(
        "Select Analysis:",
        ["Descriptive Statistics", "Normality Testing", "Regression Analysis", 
         "Correlation Analysis", "ANOVA", "Clustering Analysis"],
        index=["Descriptive Statistics", "Normality Testing", "Regression Analysis", 
               "Correlation Analysis", "ANOVA", "Clustering Analysis"].index(st.session_state.analysis_type),
        key="analysis_dropdown"
    )
    
    st.divider()
    
    st.write("### Step 2️⃣: Upload File OR Type Question")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload CSV/Excel/Image:", 
            type=["csv", "xlsx", "xls", "jpg", "jpeg", "png"],
            key="file_upload"
        )
    
    with col2:
        st.session_state.question_text = st.text_area(
            "Or type question:", 
            value=st.session_state.question_text,
            placeholder="Type your statistics question...", 
            height=120,
            key="question_input"
        )
    
    st.divider()
    
    st.write("### Step 3️⃣: Run Analysis")
    
    if st.button("🚀 RUN ANALYSIS", use_container_width=True, key="run_button"):
        if uploaded_file:
            # Process uploaded file
            if uploaded_file.name.endswith(('jpg', 'jpeg', 'png')):
                st.image(uploaded_file, use_container_width=True)
                st.info("Image uploaded - describe what you need help with")
            else:
                st.session_state.uploaded_df = read_file(uploaded_file)
                
                if st.session_state.uploaded_df is not None:
                    st.write("**Data Preview:**")
                    st.dataframe(st.session_state.uploaded_df.head(), use_container_width=True)
                    st.session_state.show_results = True
        
        elif st.session_state.question_text:
            # Process text question with AI
            api_key = get_api_key()
            if not api_key:
                st.error("❌ Admin must configure API key!")
            else:
                with st.spinner("🤖 Thinking..."):
                    answer = solve_with_ai(st.session_state.question_text, st.session_state.analysis_type, api_key)
                st.divider()
                st.markdown(answer)
                st.success("✅ Done!")
        else:
            st.error("❌ Upload file or type question!")
    
    # Show results if data is available
    if st.session_state.show_results and st.session_state.uploaded_df is not None:
        st.divider()
        st.write(f"### Analysis: {st.session_state.analysis_type}")
        
        try:
            if st.session_state.analysis_type == "Descriptive Statistics":
                descriptive_stats(st.session_state.uploaded_df)
            elif st.session_state.analysis_type == "Normality Testing":
                normality_test(st.session_state.uploaded_df)
            elif st.session_state.analysis_type == "Regression Analysis":
                regression_analysis(st.session_state.uploaded_df)
            elif st.session_state.analysis_type == "Correlation Analysis":
                correlation_analysis(st.session_state.uploaded_df)
            elif st.session_state.analysis_type == "ANOVA":
                anova_analysis(st.session_state.uploaded_df)
            elif st.session_state.analysis_type == "Clustering Analysis":
                clustering_analysis(st.session_state.uploaded_df)
        except Exception as e:
            st.error(f"Error: {str(e)}")

# ============================================================================
# HOME (LOGGED IN)
# ============================================================================

elif st.session_state.user_id and st.session_state.current_page == "home":
    st.title("📊 Data Insight Studio")
    st.write("Click **Homework** to get started!")

# ============================================================================
# ADMIN PANEL
# ============================================================================

elif st.session_state.user_id == "admin" and st.session_state.current_page == "admin":
    st.title("⚙️ Admin Panel")
    st.divider()
    
    st.write("### Configure API Key")
    
    current_key = load_api_key()
    if current_key:
        st.success(f"✅ Active: {current_key[:20]}...")
    else:
        st.warning("⚠️ No API key")
    
    st.divider()
    
    api_key_input = st.text_input(
        "API Key:",
        type="password",
        value=current_key or "",
        key="api_key_input"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Save Key", use_container_width=True, key="save_key_btn"):
            if api_key_input:
                if save_api_key(api_key_input):
                    st.session_state.api_key = api_key_input
                    st.success("✅ Saved!")
            else:
                st.error("Enter key")
    
    with col2:
        if st.button("🗑️ Delete Key", use_container_width=True, key="delete_key_btn"):
            try:
                if os.path.exists(CONFIG_FILE):
                    os.remove(CONFIG_FILE)
                st.session_state.api_key = None
                st.success("✅ Deleted!")
            except:
                st.error("Error")
    
    st.divider()
    st.write("**Get key:** https://console.anthropic.com/account/keys")
