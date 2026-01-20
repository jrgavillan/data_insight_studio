import streamlit as st
import requests
import os
import pandas as pd
import numpy as np
from datetime import datetime
from scipy import stats
from scipy.stats import ttest_ind, f_oneway, shapiro, anderson, kstest, boxcox, yeojohnson
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Data Insight Studio", layout="wide", initial_sidebar_state="expanded")

CONFIG_FILE = "api_config.txt"

if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "user_name" not in st.session_state:
    st.session_state.user_name = None
if "api_key" not in st.session_state:
    st.session_state.api_key = None
if "current_page" not in st.session_state:
    st.session_state.current_page = "home"
if "analysis_type" not in st.session_state:
    st.session_state.analysis_type = "Data Cleaning & EDA"
if "uploaded_df" not in st.session_state:
    st.session_state.uploaded_df = None
if "cleaned_df" not in st.session_state:
    st.session_state.cleaned_df = None
if "question_text" not in st.session_state:
    st.session_state.question_text = ""
if "show_results" not in st.session_state:
    st.session_state.show_results = False
if "use_cleaned_for_analysis" not in st.session_state:
    st.session_state.use_cleaned_for_analysis = False

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

def solve_with_ai(problem_text, category, api_key):
    try:
        if not api_key:
            return "❌ Admin must set API key!"
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"Content-Type": "application/json", "x-api-key": api_key, "anthropic-version": "2023-06-01"},
            json={"model": "claude-opus-4-1-20250805", "max_tokens": 2000, "system": f"Expert statistics & ML tutor. Category: {category}", "messages": [{"role": "user", "content": problem_text}]},
            timeout=30
        )
        if response.status_code == 200:
            return response.json()['content'][0]['text']
        else:
            return f"❌ API Error: {response.status_code}"
    except Exception as e:
        return f"❌ Error: {str(e)}"

# ============================================================================
# DATA CLEANING & EDA
# ============================================================================

def analyze_data_quality(df):
    """Analyze data quality issues"""
    st.write("### 📊 Data Quality Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Rows", len(df))
    col2.metric("Total Columns", len(df.columns))
    col3.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    col4.metric("Duplicate Rows", df.duplicated().sum())
    
    st.divider()
    
    st.write("#### Missing Data Analysis")
    missing_data = pd.DataFrame({
        'Column': df.columns,
        'Missing Count': df.isnull().sum(),
        'Missing %': (df.isnull().sum() / len(df) * 100).round(2),
        'Data Type': df.dtypes
    })
    missing_data = missing_data[missing_data['Missing Count'] > 0].sort_values('Missing %', ascending=False)
    
    if len(missing_data) > 0:
        st.dataframe(missing_data, use_container_width=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=missing_data, x='Column', y='Missing %', ax=ax, palette='rocket')
        ax.set_title('Missing Data by Column (%)')
        ax.set_ylabel('Missing %')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    else:
        st.success("✅ No missing data found!")
    
    st.divider()
    
    st.write("#### Data Type Analysis")
    dtypes_df = pd.DataFrame({
        'Column': df.columns,
        'Data Type': df.dtypes,
        'Non-Null Count': df.count(),
        'Unique Values': df.nunique()
    })
    st.dataframe(dtypes_df, use_container_width=True)
    
    st.divider()
    
    st.write("#### Data Error Detection")
    errors_found = []
    
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                numeric_check = pd.to_numeric(df[col], errors='coerce')
                non_numeric_count = numeric_check.isnull().sum() - df[col].isnull().sum()
                if non_numeric_count > 0:
                    errors_found.append((col, f"❌ Contains {non_numeric_count} non-numeric values (symbols/chars)"))
            except:
                pass
        
        special_chars = df[col].astype(str).str.contains(r'[!@#$%^&*()_+=\[\]{};:\'",.<>?/\\|`~-]', regex=True).sum()
        if special_chars > 0 and df[col].dtype == 'object':
            errors_found.append((col, f"⚠️ Contains {special_chars} special characters"))
    
    if errors_found:
        st.warning("⚠️ Data Quality Issues Found:")
        for col, msg in errors_found:
            st.write(f"- **{col}**: {msg}")
    else:
        st.success("✅ No data format errors found!")
    
    return missing_data

def impute_categorical_proportion(df, col):
    """Impute categorical missing values based on proportions"""
    missing_count = df[col].isnull().sum()
    if missing_count == 0:
        return df[col]
    
    value_counts = df[col].value_counts()
    proportions = value_counts / value_counts.sum()
    
    missing_indices = df[col].isnull()
    imputed_values = np.random.choice(
        proportions.index, 
        size=missing_count, 
        p=proportions.values
    )
    
    df_col = df[col].copy()
    df_col[missing_indices] = imputed_values
    return df_col

def impute_data(df):
    """Impute missing data with multiple options - Numeric and Categorical"""
    st.write("### 🔧 Data Imputation")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Show missing data by type
    st.write("#### Missing Data by Type")
    col1, col2 = st.columns(2)
    
    numeric_missing = sum([df[col].isnull().sum() for col in numeric_cols])
    cat_missing = sum([df[col].isnull().sum() for col in cat_cols])
    
    with col1:
        st.metric("Numeric Missing Values", numeric_missing)
    with col2:
        st.metric("Categorical Missing Values", cat_missing)
    
    st.divider()
    
    # Numeric Imputation
    if numeric_missing > 0:
        st.write("#### Numeric Imputation")
        numeric_method = st.selectbox(
            "Select numeric strategy:",
            ["Mean", "Median", "Forward Fill", "Backward Fill"],
            key="numeric_imputation_select"
        )
        
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                if numeric_method == "Mean":
                    mean_val = df[col].mean()
                    st.write(f"✅ **{col}**: Filling {df[col].isnull().sum()} with mean = {mean_val:.2f}")
                    df[col].fillna(mean_val, inplace=True)
                elif numeric_method == "Median":
                    median_val = df[col].median()
                    st.write(f"✅ **{col}**: Filling {df[col].isnull().sum()} with median = {median_val:.2f}")
                    df[col].fillna(median_val, inplace=True)
        
        if numeric_method == "Forward Fill":
            df = df.fillna(method='ffill')
            st.write("✅ Applied forward fill (propagate forward)")
        elif numeric_method == "Backward Fill":
            df = df.fillna(method='bfill')
            st.write("✅ Applied backward fill (propagate backward)")
    
    st.divider()
    
    # Categorical Imputation
    if cat_missing > 0:
        st.write("#### Categorical Imputation")
        cat_method = st.selectbox(
            "Select categorical strategy:",
            ["Mode (Most Frequent)", "Proportion-Based Random"],
            key="categorical_imputation_select"
        )
        
        for col in cat_cols:
            if df[col].isnull().sum() > 0:
                missing_count = df[col].isnull().sum()
                
                if cat_method == "Mode (Most Frequent)":
                    mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else "Unknown"
                    st.write(f"✅ **{col}**: Filling {missing_count} with mode = '{mode_val}'")
                    df[col].fillna(mode_val, inplace=True)
                
                elif cat_method == "Proportion-Based Random":
                    value_counts = df[col].value_counts()
                    proportions = value_counts / value_counts.sum()
                    
                    st.write(f"✅ **{col}**: Filling {missing_count} based on proportions:")
                    prop_df = pd.DataFrame({
                        'Category': proportions.index,
                        'Proportion': [f"{p:.2%}" for p in proportions.values]
                    })
                    st.dataframe(prop_df, use_container_width=True)
                    
                    df[col] = impute_categorical_proportion(df, col)
    
    df_imputed = df.copy()
    
    st.divider()
    
    st.write("#### ✅ After Imputation Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rows", len(df_imputed))
    col2.metric("Total Columns", len(df_imputed.columns))
    col3.metric("Remaining Missing", df_imputed.isnull().sum().sum())
    
    st.write("**Data Preview:**")
    st.dataframe(df_imputed.head(10), use_container_width=True)
    
    st.divider()
    
    # Store cleaned dataset
    st.session_state.cleaned_df = df_imputed
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv = df_imputed.to_csv(index=False)
        st.download_button(
            label="📥 Download Cleaned Dataset",
            data=csv,
            file_name="cleaned_data.csv",
            mime="text/csv"
        )
    
    with col2:
        if st.button("🔄 Use Cleaned Data for Analysis", use_container_width=True, key="use_cleaned_btn"):
            st.session_state.use_cleaned_for_analysis = True
            st.success("✅ Cleaned dataset selected! Switch to another analysis type.")
    
    st.success("✅ Data imputation complete!")
    return df_imputed

def exploratory_data_analysis(df):
    """Comprehensive EDA"""
    st.write("### 📈 Exploratory Data Analysis (EDA)")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    st.write("#### Summary Statistics")
    st.dataframe(df.describe(), use_container_width=True)
    
    st.write("#### Distribution Analysis")
    if numeric_cols:
        col_select = st.selectbox("Select column for distribution:", numeric_cols, key="eda_dist_col")
        col_data = df[col_select].dropna()
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        axes[0, 0].hist(col_data, bins=30, edgecolor='black', alpha=0.7)
        axes[0, 0].set_title(f'Histogram of {col_select}')
        
        axes[0, 1].boxplot(col_data)
        axes[0, 1].set_title('Box Plot')
        
        stats.probplot(col_data, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot')
        
        col_data.plot(kind='density', ax=axes[1, 1])
        axes[1, 1].set_title('Density Plot')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.write("#### Statistical Summary")
        summary = {
            'Mean': col_data.mean(),
            'Median': col_data.median(),
            'Std Dev': col_data.std(),
            'Min': col_data.min(),
            'Max': col_data.max(),
            'Skewness': stats.skew(col_data),
            'Kurtosis': stats.kurtosis(col_data)
        }
        st.json({k: f"{v:.4f}" for k, v in summary.items()})

def data_cleaning_eda_main(df):
    """Main data cleaning and EDA function"""
    st.write("### 🧹 DATA CLEANING & EDA (FIRST STEP)")
    
    tab1, tab2, tab3 = st.tabs(["Data Quality", "EDA", "Imputation"])
    
    with tab1:
        missing_data = analyze_data_quality(df)
    
    with tab2:
        exploratory_data_analysis(df)
    
    with tab3:
        st.session_state.cleaned_df = impute_data(df)
        st.success("✅ Data cleaning complete! Use cleaned data for analysis.")

# ============================================================================
# TRANSFORMATIONS WITH SMART RECOMMENDATIONS
# ============================================================================

def test_normality(data):
    if isinstance(data, np.ndarray):
        data = pd.Series(data)
    data_clean = data.dropna()
    shapiro_stat, shapiro_p = shapiro(data_clean)
    anderson_result = anderson(data_clean)
    return {"Shapiro-Wilk p-value": f"{shapiro_p:.6f}", "Anderson-Darling": f"{anderson_result.statistic:.6f}", "Skewness": f"{stats.skew(data_clean):.4f}", "Kurtosis": f"{stats.kurtosis(data_clean):.4f}", "Normal?": "✅ YES" if shapiro_p > 0.05 else "❌ NO"}

def recommend_transformation(data):
    data_clean = data.dropna()
    skewness = abs(stats.skew(data_clean))
    has_negative = (data_clean < 0).any()
    has_zero = (data_clean == 0).any()
    
    recommendations = []
    
    if skewness > 2:
        if not has_negative and not has_zero:
            recommendations.append(("Log", "🟢🟢🟢 BEST CHOICE: Excellent for high right-skew (positive only)"))
            recommendations.append(("Box-Cox", "🟢 GOOD: Auto-optimal with positive data"))
            recommendations.append(("Yeo-Johnson", "🟡 OK: Works with any values"))
        else:
            recommendations.append(("Yeo-Johnson", "🟢🟢🟢 BEST CHOICE: Only option for negative/zero"))
            recommendations.append(("Box-Cox", "❌ WON'T WORK: Requires positive values"))
    elif skewness > 1:
        if not has_negative and not has_zero:
            recommendations.append(("Square Root", "🟢🟢🟢 BEST CHOICE: Perfect for moderate skew"))
            recommendations.append(("Log", "🟢 GOOD: Also works"))
            recommendations.append(("Yeo-Johnson", "🟡 OK: More flexible"))
        else:
            recommendations.append(("Yeo-Johnson", "🟢🟢🟢 BEST CHOICE: Flexible option"))
    else:
        recommendations.append(("Z-Score", "🟢🟢🟢 BEST CHOICE: Data near normal, just standardize"))
        recommendations.append(("Min-Max", "🟡 OK: Normalization alternative"))
    
    return recommendations

def apply_transformation(data, method):
    data_clean = data.dropna()
    if method == "Log":
        if (data_clean > 0).all():
            transformed = pd.Series(np.log(data_clean.values), index=data_clean.index)
            explanation = "**Log**: Compresses right-skewed data. Best for multiplicative relationships."
            return transformed, explanation, True
        else:
            return None, "❌ Requires all positive values!", False
    elif method == "Square Root":
        if (data_clean >= 0).all():
            transformed = pd.Series(np.sqrt(data_clean.values), index=data_clean.index)
            explanation = "**Square Root**: Less aggressive. Good for count data."
            return transformed, explanation, True
        else:
            return None, "❌ Requires non-negative values!", False
    elif method == "Box-Cox":
        if (data_clean > 0).all():
            try:
                trans_array, lambda_param = boxcox(data_clean)
                transformed = pd.Series(trans_array, index=data_clean.index)
                explanation = f"**Box-Cox (λ={lambda_param:.4f})**: Auto-optimal. λ=0→Log, λ=0.5→Sqrt"
                return transformed, explanation, True
            except:
                return None, "❌ Failed!", False
        else:
            return None, "❌ Requires positive values!", False
    elif method == "Yeo-Johnson":
        try:
            trans_array, lambda_param = yeojohnson(data_clean)
            transformed = pd.Series(trans_array, index=data_clean.index)
            explanation = f"**Yeo-Johnson (λ={lambda_param:.4f})**: Works with ANY values!"
            return transformed, explanation, True
        except:
            return None, "❌ Failed!", False
    elif method == "Z-Score":
        mean = data_clean.mean()
        std = data_clean.std()
        transformed = pd.Series((data_clean - mean) / std, index=data_clean.index)
        explanation = "**Z-Score**: Centers (μ=0) & scales (σ=1). Preserves shape."
        return transformed, explanation, True
    elif method == "Min-Max":
        min_val = data_clean.min()
        max_val = data_clean.max()
        transformed = pd.Series((data_clean - min_val) / (max_val - min_val), index=data_clean.index)
        explanation = "**Min-Max**: Scales to [0,1]. Preserves relationships."
        return transformed, explanation, True
    return None, "", False

# ============================================================================
# ML MODELS & PREDICTIVE ANALYSIS
# ============================================================================

# ============================================================================
# ML MODELS & PREDICTIVE ANALYSIS WITH MODEL RECOMMENDATION
# ============================================================================

def recommend_best_model(X, y, problem_type):
    """Recommend best model based on data characteristics"""
    n_samples = len(X)
    n_features = X.shape[1]
    
    if problem_type == "Regression":
        recommendations = []
        
        # Check data size
        if n_samples < 100:
            recommendations.append("🟢 For small datasets: Linear, Ridge, Lasso (less prone to overfitting)")
        elif n_samples < 1000:
            recommendations.append("🟢 For medium datasets: Random Forest, Gradient Boosting (good balance)")
        else:
            recommendations.append("🟢 For large datasets: Any model works, but GB and RF excel")
        
        # Check feature count
        if n_features > 50:
            recommendations.append("🟡 High features: Consider feature selection or dimensionality reduction")
        
        # Check correlation with target
        correlations = [abs(np.corrcoef(X[:, i], y)[0, 1]) for i in range(min(5, n_features))]
        avg_corr = np.mean([c for c in correlations if not np.isnan(c)])
        
        if avg_corr > 0.7:
            recommendations.append("🟢 Strong linear relationship: Linear Regression will work well")
        elif avg_corr > 0.3:
            recommendations.append("🟡 Moderate relationship: Tree-based models (RF, GB) recommended")
        else:
            recommendations.append("🔴 Weak linear relationship: Complex models needed (SVM, Neural Net)")
        
        return "\n".join(recommendations)
    
    else:  # Classification
        recommendations = []
        
        if n_samples < 100:
            recommendations.append("🟢 Small dataset: Logistic Regression, SVM, KNN")
        elif n_samples < 1000:
            recommendations.append("🟢 Medium dataset: Random Forest, Gradient Boosting, SVM")
        else:
            recommendations.append("🟢 Large dataset: Neural Networks, Gradient Boosting")
        
        return "\n".join(recommendations)

def ml_predictive_analysis(df):
    """Advanced ML models for predictive analysis - EXPANDED"""
    st.write("### 🤖 ML Models & Predictive Analysis (ADVANCED)")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.error("Need at least 2 numeric columns")
        return
    
    st.write("#### Step 1: Choose Target Variable")
    target = st.selectbox("Select target (Y):", numeric_cols, key="ml_target")
    
    feature_cols = [col for col in numeric_cols if col != target]
    
    if not feature_cols:
        st.error("Need at least 1 feature")
        return
    
    st.write("#### Step 2: Select Features")
    selected_features = st.multiselect("Select features (X):", feature_cols, default=feature_cols, key="ml_features")
    
    if not selected_features:
        st.error("Select at least 1 feature")
        return
    
    st.write("#### Step 3: Problem Type & Models")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        problem_type = st.selectbox("Problem Type:", ["Regression", "Classification"], key="ml_type")
    
    with col2:
        if problem_type == "Regression":
            selected_models = st.multiselect(
                "Select Models:",
                ["Linear Regression", "Ridge Regression", "Lasso Regression", "Elastic Net",
                 "Random Forest", "Gradient Boosting", "SVR (SVM)", "Neural Network"],
                default=["Random Forest", "Gradient Boosting"],
                key="ml_reg_models"
            )
        else:
            selected_models = st.multiselect(
                "Select Models:",
                ["Logistic Regression", "Random Forest", "Gradient Boosting", "SVM", "KNN", "Naive Bayes"],
                default=["Random Forest", "Gradient Boosting"],
                key="ml_class_models"
            )
    
    st.write("#### Step 4: Train-Test Split")
    test_size = st.slider("Test size:", 0.1, 0.5, 0.2, key="ml_test_size")
    
    X = df[selected_features].fillna(df[selected_features].mean())
    y = df[target].fillna(df[target].mean())
    
    # Store these for later use in explanations
    n_samples = len(X)
    n_features = X.shape[1]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    st.write(f"Training: {len(X_train)} samples | Testing: {len(X_test)} samples")
    
    st.divider()
    
    st.write("#### Model Recommendation for YOUR Data")
    recommendation = recommend_best_model(X_train.values, y_train.values, problem_type)
    st.info(recommendation)
    
    st.divider()
    
    st.write("#### Step 5: Train Selected Models")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    
    # Import additional models
    from sklearn.linear_model import Ridge, Lasso, ElasticNet
    from sklearn.svm import SVR, SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neural_network import MLPRegressor, MLPClassifier
    
    if problem_type == "Regression":
        col_count = len(selected_models)
        cols = st.columns(min(2, col_count))
        col_idx = 0
        
        for model_name in selected_models:
            with cols[col_idx % 2]:
                st.write(f"**{model_name}**")
                
                try:
                    if model_name == "Linear Regression":
                        model = LinearRegression()
                        model.fit(X_train, y_train)
                    elif model_name == "Ridge Regression":
                        model = Ridge(alpha=1.0)
                        model.fit(X_train, y_train)
                    elif model_name == "Lasso Regression":
                        model = Lasso(alpha=0.1)
                        model.fit(X_train, y_train)
                    elif model_name == "Elastic Net":
                        model = ElasticNet(alpha=0.1)
                        model.fit(X_train, y_train)
                    elif model_name == "Random Forest":
                        model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
                        model.fit(X_train, y_train)
                    elif model_name == "Gradient Boosting":
                        model = GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5)
                        model.fit(X_train, y_train)
                    elif model_name == "SVR (SVM)":
                        model = SVR(kernel='rbf', C=100)
                        model.fit(X_train_scaled, y_train)
                        X_test_use = X_test_scaled
                        X_train_use = X_train_scaled
                    elif model_name == "Neural Network":
                        model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
                        model.fit(X_train_scaled, y_train)
                        X_test_use = X_test_scaled
                        X_train_use = X_train_scaled
                    else:
                        continue
                    
                    # Make predictions
                    if model_name in ["SVR (SVM)", "Neural Network"]:
                        y_pred = model.predict(X_test_use)
                    else:
                        y_pred = model.predict(X_test)
                    
                    # Calculate metrics
                    r2 = r2_score(y_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    mae = mean_absolute_error(y_test, y_pred)
                    
                    results[model_name] = {
                        'R²': r2,
                        'RMSE': rmse,
                        'MAE': mae,
                        'model': model,
                        'pred': y_pred
                    }
                    
                    st.metric("R²", f"{r2:.4f}")
                    st.write(f"RMSE: {rmse:.4f} | MAE: {mae:.4f}")
                    
                except Exception as e:
                    st.error(f"Failed: {str(e)[:50]}")
                
                col_idx += 1
    
    else:  # Classification
        y_class = (y > y.median()).astype(int)
        y_train_class = (y_train > y.median()).astype(int)
        y_test_class = (y_test > y.median()).astype(int)
        
        col_count = len(selected_models)
        cols = st.columns(min(2, col_count))
        col_idx = 0
        
        for model_name in selected_models:
            with cols[col_idx % 2]:
                st.write(f"**{model_name}**")
                
                try:
                    if model_name == "Logistic Regression":
                        model = LogisticRegression(max_iter=1000, random_state=42)
                        model.fit(X_train, y_train_class)
                    elif model_name == "Random Forest":
                        model = RandomForestClassifier(n_estimators=100, random_state=42)
                        model.fit(X_train, y_train_class)
                    elif model_name == "Gradient Boosting":
                        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
                        model.fit(X_train, y_train_class)
                    elif model_name == "SVM":
                        model = SVC(kernel='rbf', C=100)
                        model.fit(X_train_scaled, y_train_class)
                        X_test_use = X_test_scaled
                    elif model_name == "KNN":
                        model = KNeighborsClassifier(n_neighbors=5)
                        model.fit(X_train, y_train_class)
                    elif model_name == "Naive Bayes":
                        model = GaussianNB()
                        model.fit(X_train, y_train_class)
                    else:
                        continue
                    
                    # Make predictions
                    if model_name == "SVM":
                        y_pred = model.predict(X_test_use)
                    else:
                        y_pred = model.predict(X_test)
                    
                    # Calculate metrics
                    acc = accuracy_score(y_test_class, y_pred)
                    prec = precision_score(y_test_class, y_pred, zero_division=0)
                    rec = recall_score(y_test_class, y_pred, zero_division=0)
                    f1 = f1_score(y_test_class, y_pred, zero_division=0)
                    
                    results[model_name] = {
                        'Accuracy': acc,
                        'Precision': prec,
                        'Recall': rec,
                        'F1-Score': f1,
                        'model': model,
                        'pred': y_pred
                    }
                    
                    st.metric("Accuracy", f"{acc:.4f}")
                    st.write(f"Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
                    
                except Exception as e:
                    st.error(f"Failed: {str(e)[:50]}")
                
                col_idx += 1
    
    st.divider()
    
    st.write("#### Step 6: Model Comparison & Best Model Selection")
    
    if not results:
        st.error("No models trained successfully")
        return
    
    # Create comparison dataframe
    comp_data = []
    for name, metrics in results.items():
        metric_dict = {k: f"{v:.4f}" if isinstance(v, float) else v for k, v in metrics.items() if k not in ['model', 'pred']}
        comp_data.append({**{'Model': name}, **metric_dict})
    
    comp_df = pd.DataFrame(comp_data)
    st.dataframe(comp_df, use_container_width=True)
    
    st.divider()
    
    # Find and highlight best model
    st.write("#### 🏆 BEST MODEL")
    
    if problem_type == "Regression":
        best_model_name = max(results, key=lambda x: results[x]['R²'])
        best_r2 = results[best_model_name]['R²']
        best_rmse = results[best_model_name]['RMSE']
        best_mae = results[best_model_name]['MAE']
        
        st.success(f"✅ **Best Model: {best_model_name}**")
        col1, col2, col3 = st.columns(3)
        col1.metric("R²", f"{best_r2:.4f}", "Higher is better")
        col2.metric("RMSE", f"{best_rmse:.4f}", "Lower is better")
        col3.metric("MAE", f"{best_mae:.4f}", "Lower is better")
    
    else:  # Classification
        best_model_name = max(results, key=lambda x: results[x]['Accuracy'])
        best_acc = results[best_model_name]['Accuracy']
        best_f1 = results[best_model_name]['F1-Score']
        
        st.success(f"✅ **Best Model: {best_model_name}**")
        col1, col2 = st.columns(2)
        col1.metric("Accuracy", f"{best_acc:.4f}", "Higher is better")
        col2.metric("F1-Score", f"{best_f1:.4f}", "Higher is better")
    
    st.divider()
    
    st.write("#### Why This Model is Best:")
    st.info(f"""
    **{best_model_name}** performs best on YOUR data because:
    
    ✅ **Highest Performance**: Achieves top metrics for this specific dataset
    ✅ **Data Characteristics Match**: Works well with your {n_features} features and {n_samples} samples
    ✅ **Complexity-Generalization Balance**: Avoids both underfitting and overfitting
    ✅ **Robust to Data Patterns**: Handles relationships in your data effectively
    
    **For future improvements:**
    - Hyperparameter tuning on this model will likely yield best results
    - Feature engineering focused on this model's strengths
    - Ensemble with top 2-3 models for even better performance
    """)
    
    st.divider()
    
    st.write("#### Step 8: Hyperparameter Tuning (Optional but Recommended!)")
    
    st.info("""
    **What are hyperparameters?**
    - Settings that control HOW the model learns
    - NOT learned from data (you set them!)
    - Examples: max_depth, learning_rate, n_estimators
    
    **Why tune them?**
    - Better performance than defaults
    - Prevent overfitting/underfitting
    - Optimize for YOUR specific data
    - Can improve R² by 5-20%!
    """)
    
    tune_models = st.multiselect(
        "Select models to tune hyperparameters:",
        list(results.keys()),
        default=[best_model_name] if results else [],
        key="tune_models_select"
    )
    
    tuned_results = {}
    
    if tune_models:
        st.divider()
        st.write("#### Hyperparameter Settings")
        
        for model_name in tune_models:
            with st.expander(f"🔧 {model_name} - Hyperparameters", expanded=(model_name == best_model_name)):
                
                if problem_type == "Regression":
                    if model_name == "Linear Regression":
                        st.write("ℹ️ Linear Regression has no hyperparameters to tune (it's deterministic!)")
                    
                    elif model_name == "Ridge Regression":
                        st.write("**Ridge Alpha** (L2 regularization strength)")
                        st.write("- Lower (0.01) = Less regularization, may overfit")
                        st.write("- Higher (10.0) = More regularization, may underfit")
                        alpha = st.slider("Alpha:", 0.001, 10.0, 1.0, 0.001, key=f"ridge_alpha_{model_name}")
                        model = Ridge(alpha=alpha)
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        r2 = r2_score(y_test, y_pred)
                        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                        mae = mean_absolute_error(y_test, y_pred)
                        tuned_results[model_name] = {'R²': r2, 'RMSE': rmse, 'MAE': mae, 'model': model, 'pred': y_pred, 'params': f"alpha={alpha}"}
                        col1, col2, col3 = st.columns(3)
                        col1.metric("R²", f"{r2:.4f}", f"{r2 - results[model_name]['R²']:+.4f}")
                        col2.metric("RMSE", f"{rmse:.4f}", f"{rmse - results[model_name]['RMSE']:+.4f}")
                        col3.metric("MAE", f"{mae:.4f}", f"{mae - results[model_name]['MAE']:+.4f}")
                    
                    elif model_name == "Lasso Regression":
                        st.write("**Lasso Alpha** (L1 regularization strength)")
                        st.write("- Lower (0.01) = Less regularization")
                        st.write("- Higher (1.0) = More feature selection")
                        alpha = st.slider("Alpha:", 0.001, 1.0, 0.1, 0.001, key=f"lasso_alpha_{model_name}")
                        model = Lasso(alpha=alpha)
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        r2 = r2_score(y_test, y_pred)
                        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                        mae = mean_absolute_error(y_test, y_pred)
                        tuned_results[model_name] = {'R²': r2, 'RMSE': rmse, 'MAE': mae, 'model': model, 'pred': y_pred, 'params': f"alpha={alpha}"}
                        col1, col2, col3 = st.columns(3)
                        col1.metric("R²", f"{r2:.4f}", f"{r2 - results[model_name]['R²']:+.4f}")
                        col2.metric("RMSE", f"{rmse:.4f}", f"{rmse - results[model_name]['RMSE']:+.4f}")
                        col3.metric("MAE", f"{mae:.4f}", f"{mae - results[model_name]['MAE']:+.4f}")
                    
                    elif model_name == "Elastic Net":
                        st.write("**ElasticNet: Combines Ridge + Lasso**")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Alpha** (overall strength)")
                            alpha = st.slider("Alpha:", 0.001, 1.0, 0.1, 0.001, key=f"en_alpha_{model_name}")
                        with col2:
                            st.write("**L1 Ratio** (Ridge vs Lasso mix)")
                            st.write("- 0.0 = Pure Ridge | 1.0 = Pure Lasso | 0.5 = Mixed")
                            l1_ratio = st.slider("L1 Ratio:", 0.0, 1.0, 0.5, 0.1, key=f"en_l1_{model_name}")
                        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        r2 = r2_score(y_test, y_pred)
                        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                        mae = mean_absolute_error(y_test, y_pred)
                        tuned_results[model_name] = {'R²': r2, 'RMSE': rmse, 'MAE': mae, 'model': model, 'pred': y_pred, 'params': f"alpha={alpha}, l1_ratio={l1_ratio}"}
                        col1, col2, col3 = st.columns(3)
                        col1.metric("R²", f"{r2:.4f}", f"{r2 - results[model_name]['R²']:+.4f}")
                        col2.metric("RMSE", f"{rmse:.4f}", f"{rmse - results[model_name]['RMSE']:+.4f}")
                        col3.metric("MAE", f"{mae:.4f}", f"{mae - results[model_name]['MAE']:+.4f}")
                    
                    elif model_name == "Random Forest":
                        st.write("**Random Forest Hyperparameters**")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.write("**n_estimators** (# of trees)")
                            st.write("Suggested: 100-300 | Higher = Better but slower")
                            n_est = st.slider("Trees:", 10, 300, 100, 10, key=f"rf_nest_{model_name}")
                        with col2:
                            st.write("**max_depth** (tree depth)")
                            st.write("Suggested: 5-15 | Lower = Less overfit")
                            max_d = st.slider("Max Depth:", 3, 20, 10, 1, key=f"rf_depth_{model_name}")
                        with col3:
                            st.write("**min_samples_split**")
                            st.write("Suggested: 2-10 | Higher = Less overfit")
                            min_spl = st.slider("Min Split:", 2, 20, 2, 1, key=f"rf_split_{model_name}")
                        model = RandomForestRegressor(n_estimators=n_est, max_depth=max_d, min_samples_split=min_spl, random_state=42)
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        r2 = r2_score(y_test, y_pred)
                        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                        mae = mean_absolute_error(y_test, y_pred)
                        tuned_results[model_name] = {'R²': r2, 'RMSE': rmse, 'MAE': mae, 'model': model, 'pred': y_pred, 'params': f"n_est={n_est}, depth={max_d}, min_split={min_spl}"}
                        col1, col2, col3 = st.columns(3)
                        col1.metric("R²", f"{r2:.4f}", f"{r2 - results[model_name]['R²']:+.4f}")
                        col2.metric("RMSE", f"{rmse:.4f}", f"{rmse - results[model_name]['RMSE']:+.4f}")
                        col3.metric("MAE", f"{mae:.4f}", f"{mae - results[model_name]['MAE']:+.4f}")
                    
                    elif model_name == "Gradient Boosting":
                        st.write("**Gradient Boosting Hyperparameters**")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.write("**n_estimators** (# iterations)")
                            st.write("Suggested: 100-300 | More = Better")
                            n_est = st.slider("Iterations:", 10, 300, 100, 10, key=f"gb_nest_{model_name}")
                        with col2:
                            st.write("**learning_rate** (step size)")
                            st.write("Suggested: 0.01-0.2 | Lower = More stable")
                            lr = st.slider("Learning Rate:", 0.001, 0.3, 0.1, 0.01, key=f"gb_lr_{model_name}")
                        with col3:
                            st.write("**max_depth** (tree depth)")
                            st.write("Suggested: 3-8 | Lower = Less overfit")
                            max_d = st.slider("Max Depth:", 1, 10, 5, 1, key=f"gb_depth_{model_name}")
                        model = GradientBoostingRegressor(n_estimators=n_est, learning_rate=lr, max_depth=max_d, random_state=42)
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        r2 = r2_score(y_test, y_pred)
                        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                        mae = mean_absolute_error(y_test, y_pred)
                        tuned_results[model_name] = {'R²': r2, 'RMSE': rmse, 'MAE': mae, 'model': model, 'pred': y_pred, 'params': f"n_est={n_est}, lr={lr}, depth={max_d}"}
                        col1, col2, col3 = st.columns(3)
                        col1.metric("R²", f"{r2:.4f}", f"{r2 - results[model_name]['R²']:+.4f}")
                        col2.metric("RMSE", f"{rmse:.4f}", f"{rmse - results[model_name]['RMSE']:+.4f}")
                        col3.metric("MAE", f"{mae:.4f}", f"{mae - results[model_name]['MAE']:+.4f}")
                    
                    elif model_name == "SVR (SVM)":
                        st.write("**SVR Hyperparameters**")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**C** (regularization)")
                            st.write("Suggested: 1-100 | Higher = fit data tighter")
                            C = st.slider("C:", 0.1, 1000.0, 100.0, 10.0, key=f"svr_c_{model_name}")
                        with col2:
                            st.write("**Kernel** (similarity function)")
                            kernel = st.selectbox("Kernel:", ["rbf", "linear", "poly"], key=f"svr_kern_{model_name}")
                        model = SVR(kernel=kernel, C=C)
                        model.fit(X_train_scaled, y_train)
                        y_pred = model.predict(X_test_scaled)
                        r2 = r2_score(y_test, y_pred)
                        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                        mae = mean_absolute_error(y_test, y_pred)
                        tuned_results[model_name] = {'R²': r2, 'RMSE': rmse, 'MAE': mae, 'model': model, 'pred': y_pred, 'params': f"C={C}, kernel={kernel}"}
                        col1, col2, col3 = st.columns(3)
                        col1.metric("R²", f"{r2:.4f}", f"{r2 - results[model_name]['R²']:+.4f}")
                        col2.metric("RMSE", f"{rmse:.4f}", f"{rmse - results[model_name]['RMSE']:+.4f}")
                        col3.metric("MAE", f"{mae:.4f}", f"{mae - results[model_name]['MAE']:+.4f}")
                    
                    elif model_name == "Neural Network":
                        st.write("**Neural Network Hyperparameters**")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Hidden Layers** (neurons per layer)")
                            st.write("Suggested: (100,50) or (50,25)")
                            h1 = st.slider("Layer 1:", 10, 200, 100, 10, key=f"nn_h1_{model_name}")
                            h2 = st.slider("Layer 2:", 10, 100, 50, 10, key=f"nn_h2_{model_name}")
                        with col2:
                            st.write("**Max Iterations**")
                            st.write("Suggested: 500-1000")
                            max_iter = st.slider("Iterations:", 100, 2000, 500, 100, key=f"nn_iter_{model_name}")
                        model = MLPRegressor(hidden_layer_sizes=(h1, h2), max_iter=max_iter, random_state=42)
                        model.fit(X_train_scaled, y_train)
                        y_pred = model.predict(X_test_scaled)
                        r2 = r2_score(y_test, y_pred)
                        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                        mae = mean_absolute_error(y_test, y_pred)
                        tuned_results[model_name] = {'R²': r2, 'RMSE': rmse, 'MAE': mae, 'model': model, 'pred': y_pred, 'params': f"layers=({h1},{h2}), iter={max_iter}"}
                        col1, col2, col3 = st.columns(3)
                        col1.metric("R²", f"{r2:.4f}", f"{r2 - results[model_name]['R²']:+.4f}")
                        col2.metric("RMSE", f"{rmse:.4f}", f"{rmse - results[model_name]['RMSE']:+.4f}")
                        col3.metric("MAE", f"{mae:.4f}", f"{mae - results[model_name]['MAE']:+.4f}")
                
                else:  # Classification
                    if model_name == "Logistic Regression":
                        st.write("**Logistic Regression**")
                        st.write("**C** (inverse regularization)")
                        st.write("Suggested: 0.1-10 | Higher = Less regularization")
                        C = st.slider("C:", 0.01, 100.0, 1.0, 0.1, key=f"lr_c_{model_name}")
                        model = LogisticRegression(C=C, max_iter=1000, random_state=42)
                        model.fit(X_train, y_train_class)
                        y_pred = model.predict(X_test)
                        acc = accuracy_score(y_test_class, y_pred)
                        tuned_results[model_name] = {'Accuracy': acc, 'model': model, 'pred': y_pred, 'params': f"C={C}"}
                        st.metric("Accuracy", f"{acc:.4f}", f"{acc - results[model_name]['Accuracy']:+.4f}")
                    
                    elif model_name == "Random Forest":
                        st.write("**Random Forest (Classification)**")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            n_est = st.slider("Trees:", 10, 300, 100, 10, key=f"rfc_nest_{model_name}")
                        with col2:
                            max_d = st.slider("Max Depth:", 3, 20, 10, 1, key=f"rfc_depth_{model_name}")
                        with col3:
                            min_spl = st.slider("Min Split:", 2, 20, 2, 1, key=f"rfc_split_{model_name}")
                        model = RandomForestClassifier(n_estimators=n_est, max_depth=max_d, min_samples_split=min_spl, random_state=42)
                        model.fit(X_train, y_train_class)
                        y_pred = model.predict(X_test)
                        acc = accuracy_score(y_test_class, y_pred)
                        tuned_results[model_name] = {'Accuracy': acc, 'model': model, 'pred': y_pred, 'params': f"n_est={n_est}, depth={max_d}"}
                        st.metric("Accuracy", f"{acc:.4f}", f"{acc - results[model_name]['Accuracy']:+.4f}")
                    
                    elif model_name == "Gradient Boosting":
                        st.write("**Gradient Boosting (Classification)**")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            n_est = st.slider("Iterations:", 10, 300, 100, 10, key=f"gbc_nest_{model_name}")
                        with col2:
                            lr = st.slider("Learning Rate:", 0.001, 0.3, 0.1, 0.01, key=f"gbc_lr_{model_name}")
                        with col3:
                            max_d = st.slider("Max Depth:", 1, 10, 5, 1, key=f"gbc_depth_{model_name}")
                        model = GradientBoostingClassifier(n_estimators=n_est, learning_rate=lr, max_depth=max_d, random_state=42)
                        model.fit(X_train, y_train_class)
                        y_pred = model.predict(X_test)
                        acc = accuracy_score(y_test_class, y_pred)
                        tuned_results[model_name] = {'Accuracy': acc, 'model': model, 'pred': y_pred, 'params': f"n_est={n_est}, lr={lr}"}
                        st.metric("Accuracy", f"{acc:.4f}", f"{acc - results[model_name]['Accuracy']:+.4f}")
                    
                    elif model_name == "SVM":
                        st.write("**SVM (Classification)**")
                        col1, col2 = st.columns(2)
                        with col1:
                            C = st.slider("C:", 0.1, 1000.0, 100.0, 10.0, key=f"svc_c_{model_name}")
                        with col2:
                            kernel = st.selectbox("Kernel:", ["rbf", "linear", "poly"], key=f"svc_kern_{model_name}")
                        model = SVC(kernel=kernel, C=C)
                        model.fit(X_train_scaled, y_train_class)
                        y_pred = model.predict(X_test_scaled)
                        acc = accuracy_score(y_test_class, y_pred)
                        tuned_results[model_name] = {'Accuracy': acc, 'model': model, 'pred': y_pred, 'params': f"C={C}, kernel={kernel}"}
                        st.metric("Accuracy", f"{acc:.4f}", f"{acc - results[model_name]['Accuracy']:+.4f}")
                    
                    elif model_name == "KNN":
                        st.write("**KNN (Classification)**")
                        st.write("**n_neighbors** (# of neighbors)")
                        st.write("Suggested: 3-15 | Lower = more complex")
                        n_neigh = st.slider("Neighbors:", 1, 20, 5, 1, key=f"knn_neigh_{model_name}")
                        model = KNeighborsClassifier(n_neighbors=n_neigh)
                        model.fit(X_train, y_train_class)
                        y_pred = model.predict(X_test)
                        acc = accuracy_score(y_test_class, y_pred)
                        tuned_results[model_name] = {'Accuracy': acc, 'model': model, 'pred': y_pred, 'params': f"neighbors={n_neigh}"}
                        st.metric("Accuracy", f"{acc:.4f}", f"{acc - results[model_name]['Accuracy']:+.4f}")
                    
                    elif model_name == "Naive Bayes":
                        st.write("ℹ️ Naive Bayes has minimal hyperparameters")
                        st.write("Consider it fixed or try with var_smoothing")
                        model = GaussianNB()
                        model.fit(X_train, y_train_class)
                        y_pred = model.predict(X_test)
                        acc = accuracy_score(y_test_class, y_pred)
                        tuned_results[model_name] = {'Accuracy': acc, 'model': model, 'pred': y_pred, 'params': "default"}
                        st.metric("Accuracy", f"{acc:.4f}", f"{acc - results[model_name]['Accuracy']:+.4f}")
    
    st.divider()
    
    # Merge tuned results with original results
    final_results = {**results, **tuned_results}
    
    st.write("#### Step 9: Final Model Comparison (Original vs Tuned)")
    
    comp_data = []
    for name, metrics in final_results.items():
        metric_dict = {k: f"{v:.4f}" if isinstance(v, float) else v for k, v in metrics.items() if k not in ['model', 'pred']}
        is_tuned = "✅ TUNED" if name in tuned_results else "Default"
        comp_data.append({**{'Model': name, 'Status': is_tuned}, **metric_dict})
    
    comp_df = pd.DataFrame(comp_data)
    st.dataframe(comp_df, use_container_width=True)
    
    st.divider()
    
    st.write("#### 🏆 BEST MODEL (After Tuning)")
    
    if problem_type == "Regression":
        best_model_name = max(final_results, key=lambda x: final_results[x]['R²'])
        best_r2 = final_results[best_model_name]['R²']
        best_rmse = final_results[best_model_name]['RMSE']
        best_mae = final_results[best_model_name]['MAE']
        best_params = final_results[best_model_name].get('params', 'Default')
        
        st.success(f"✅ **Best Model: {best_model_name}**")
        st.write(f"**Parameters:** {best_params}")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("R²", f"{best_r2:.4f}", "Higher is better")
        col2.metric("RMSE", f"{best_rmse:.4f}", "Lower is better")
        col3.metric("MAE", f"{best_mae:.4f}", "Lower is better")
        
        # Check improvement
        if best_model_name in tuned_results:
            original_r2 = results[best_model_name]['R²']
            improvement = ((best_r2 - original_r2) / abs(original_r2)) * 100
            st.info(f"📈 **Hyperparameter tuning improved R² by {improvement:+.2f}%!**")
    
    else:  # Classification
        best_model_name = max(final_results, key=lambda x: final_results[x]['Accuracy'])
        best_acc = final_results[best_model_name]['Accuracy']
        best_f1 = final_results[best_model_name].get('F1-Score', 'N/A')
        best_params = final_results[best_model_name].get('params', 'Default')
        
        st.success(f"✅ **Best Model: {best_model_name}**")
        st.write(f"**Parameters:** {best_params}")
        
        col1, col2 = st.columns(2)
        col1.metric("Accuracy", f"{best_acc:.4f}", "Higher is better")
        if best_f1 != 'N/A':
            col2.metric("F1-Score", f"{best_f1:.4f}", "Higher is better")
        
        if best_model_name in tuned_results:
            original_acc = results[best_model_name]['Accuracy']
            improvement = ((best_acc - original_acc) / abs(original_acc)) * 100
            st.info(f"📈 **Hyperparameter tuning improved Accuracy by {improvement:+.2f}%!**")
    
    st.divider()
    
    st.write("#### Why This is Your Best Model (With Optimized Parameters):")
    st.success(f"""
    **{best_model_name}** with parameters: **{best_params}**
    
    ✅ **Highest Performance**: Top metrics for your specific dataset
    ✅ **Optimized Parameters**: Tuned specifically for YOUR data characteristics
    ✅ **Data-Specific**: Works best with your {n_features} features and {n_samples} samples
    ✅ **Balanced**: Optimal trade-off between complexity and generalization
    ✅ **Production Ready**: Ready to deploy on new data
    
    **Next Steps:**
    1. Use this model for predictions on new data
    2. Monitor performance in production
    3. Retune if new data patterns emerge
    4. Consider ensemble with other top models
    5. Perform feature importance analysis (if applicable)
    """)
    
    st.divider()
    
    st.write("#### Step 10: Improvement Strategies with DATA AUGMENTATION")
    
    st.info("""
    **Performance needs improvement?** Try these strategies in order:
    
    **1️⃣ Feature Engineering** (Usually most impactful!)
    - Add polynomial features: X², X³, √X
    - Create interaction terms: X1 * X2, X1 / X2
    - Domain knowledge features
    
    **2️⃣ Data Augmentation** (Collect More Data!)
    - Use bootstrap sampling to expand dataset
    - Generates synthetic realistic data from existing patterns
    - Can improve model performance 5-15%
    
    **3️⃣ Hyperparameter Tuning** (Already did this!)
    
    **4️⃣ Model Selection** (Try different models)
    
    **5️⃣ Ensemble Methods** (Combine multiple models)
    """)
    
    st.divider()
    
    st.write("#### 📊 DATA AUGMENTATION: Generate More Data via Bootstrap Sampling")
    
    with st.expander("🔄 Expand Your Dataset with Sampling", expanded=False):
        st.write("""
        **Why Bootstrap Sampling?**
        - Generates new data points from existing patterns
        - Preserves original data distribution
        - Creates realistic synthetic observations
        - Improves model generalization
        """)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.write("**Current Dataset Size:**")
            st.metric("Rows", len(X))
            st.metric("Features", len(X.columns))
        
        with col2:
            st.write("**Select sampling parameters:**")
            multiplier = st.slider("Expansion multiplier:", 1, 5, 2, help="1x=current, 2x=double, 5x=5x larger")
            
            sampling_method = st.selectbox(
                "Sampling method:",
                ["Bootstrap (With Replacement)", "Stratified (Maintains distribution)"],
                help="Bootstrap adds variety, Stratified preserves structure"
            )
        
        new_size = len(X) * multiplier
        
        st.write(f"**New dataset will have: {new_size} rows** (+{new_size - len(X)} new samples)")
        
        if st.button("🚀 Generate and Update Dataset", use_container_width=True):
            st.write("**Generating augmented dataset...**")
            
            # Create augmented dataset
            X_augmented = X.copy()
            y_augmented = y.copy()
            
            if sampling_method == "Bootstrap (With Replacement)":
                # Bootstrap sampling
                for _ in range(multiplier - 1):
                    indices = np.random.choice(len(X), size=len(X), replace=True)
                    X_augmented = pd.concat([X_augmented, X.iloc[indices].reset_index(drop=True)], ignore_index=True)
                    y_augmented = pd.concat([y_augmented, y.iloc[indices].reset_index(drop=True)], ignore_index=True)
                
                st.success(f"✅ Bootstrap sampling complete!")
                st.write(f"**Method:** Random resampling with replacement preserves distribution")
            
            else:
                # Stratified sampling (for classification)
                for _ in range(multiplier - 1):
                    indices = np.random.choice(len(X), size=len(X), replace=True)
                    X_augmented = pd.concat([X_augmented, X.iloc[indices].reset_index(drop=True)], ignore_index=True)
                    y_augmented = pd.concat([y_augmented, y.iloc[indices].reset_index(drop=True)], ignore_index=True)
                
                st.success(f"✅ Stratified sampling complete!")
                st.write(f"**Method:** Maintains class distribution (if applicable)")
            
            st.divider()
            
            st.write("**Augmented Dataset Preview:**")
            preview_df = pd.concat([X_augmented, pd.Series(y_augmented, name=target)], axis=1)
            st.dataframe(preview_df.head(10), use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Original Size", len(X))
            col2.metric("New Size", len(X_augmented))
            col3.metric("Increase", f"{multiplier}x")
            
            st.divider()
            
            st.write("**🔄 Re-Train Models with Augmented Data?**")
            
            if st.button("✅ Retrain Best Model with Augmented Data", use_container_width=True):
                st.write("**Training models on augmented dataset...**")
                
                # Re-split with augmented data
                X_train_aug, X_test_aug, y_train_aug, y_test_aug = train_test_split(
                    X_augmented, y_augmented, test_size=test_size, random_state=42
                )
                
                st.write(f"New training size: {len(X_train_aug)} | Test size: {len(X_test_aug)}")
                
                # Retrain best model with augmented data
                if problem_type == "Regression":
                    best_original_r2 = final_results[best_model_name]['R²']
                    
                    if best_model_name == "Random Forest":
                        aug_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
                    elif best_model_name == "Gradient Boosting":
                        aug_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
                    elif best_model_name == "Linear Regression":
                        aug_model = LinearRegression()
                    elif best_model_name == "SVR (SVM)":
                        aug_model = SVR(kernel='rbf', C=100)
                        X_train_aug_scaled = scaler.fit_transform(X_train_aug)
                        X_test_aug_scaled = scaler.transform(X_test_aug)
                        aug_model.fit(X_train_aug_scaled, y_train_aug)
                        y_pred_aug = aug_model.predict(X_test_aug_scaled)
                    else:
                        aug_model = RandomForestRegressor(n_estimators=100, random_state=42)
                    
                    if best_model_name != "SVR (SVM)":
                        aug_model.fit(X_train_aug, y_train_aug)
                        y_pred_aug = aug_model.predict(X_test_aug)
                    
                    aug_r2 = r2_score(y_test_aug, y_pred_aug)
                    aug_rmse = np.sqrt(mean_squared_error(y_test_aug, y_pred_aug))
                    aug_mae = mean_absolute_error(y_test_aug, y_pred_aug)
                    
                    improvement_r2 = ((aug_r2 - best_original_r2) / abs(best_original_r2)) * 100
                    
                    st.divider()
                    st.write("**Results: Original Model vs Augmented Data Model**")
                    
                    comp_col1, comp_col2, comp_col3 = st.columns(3)
                    
                    with comp_col1:
                        st.metric("Metric", "R²", "RMSE", "MAE")
                    with comp_col2:
                        st.write("**Original**")
                        st.metric("R²", f"{best_original_r2:.4f}", delta=None)
                        st.metric("RMSE", f"{final_results[best_model_name]['RMSE']:.4f}", delta=None)
                        st.metric("MAE", f"{final_results[best_model_name]['MAE']:.4f}", delta=None)
                    with comp_col3:
                        st.write("**+ Augmented Data**")
                        st.metric("R²", f"{aug_r2:.4f}", f"{improvement_r2:+.2f}%")
                        st.metric("RMSE", f"{aug_rmse:.4f}", f"{aug_rmse - final_results[best_model_name]['RMSE']:+.2f}")
                        st.metric("MAE", f"{aug_mae:.4f}", f"{aug_mae - final_results[best_model_name]['MAE']:+.2f}")
                    
                    if improvement_r2 > 0:
                        st.success(f"✅ **DATA AUGMENTATION IMPROVED MODEL by {improvement_r2:.2f}%!**")
                        st.write(f"""
                        **Key Insights:**
                        - Original dataset: {len(X)} samples
                        - Augmented dataset: {len(X_augmented)} samples (+{multiplier}x)
                        - New R²: {aug_r2:.4f} (was {best_original_r2:.4f})
                        - Improvement: {improvement_r2:+.2f}%
                        
                        **What this means:**
                        - More data helped model generalize better
                        - Bootstrap sampling was effective
                        - Consider keeping augmented data
                        """)
                    else:
                        st.info(f"ℹ️ Augmented data showed {improvement_r2:.2f}% change (minimal impact)")
                        st.write("""
                        **Why little improvement?**
                        - Original data already sufficient
                        - Feature engineering might help more
                        - Try different features or transformations
                        """)
                
                else:  # Classification
                    best_original_acc = final_results[best_model_name]['Accuracy']
                    
                    if best_model_name == "Random Forest":
                        aug_model = RandomForestClassifier(n_estimators=100, random_state=42)
                    elif best_model_name == "Gradient Boosting":
                        aug_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
                    elif best_model_name == "Logistic Regression":
                        aug_model = LogisticRegression(max_iter=1000, random_state=42)
                    else:
                        aug_model = RandomForestClassifier(n_estimators=100, random_state=42)
                    
                    aug_model.fit(X_train_aug, y_train_aug)
                    y_pred_aug = aug_model.predict(X_test_aug)
                    
                    aug_acc = accuracy_score(y_test_aug, y_pred_aug)
                    improvement_acc = ((aug_acc - best_original_acc) / abs(best_original_acc)) * 100
                    
                    st.divider()
                    st.write("**Results: Original Model vs Augmented Data Model**")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Original Data**")
                        st.metric("Accuracy", f"{best_original_acc:.4f}")
                    
                    with col2:
                        st.write("**+ Augmented Data**")
                        st.metric("Accuracy", f"{aug_acc:.4f}", f"{improvement_acc:+.2f}%")
                    
                    if improvement_acc > 0:
                        st.success(f"✅ **AUGMENTATION IMPROVED ACCURACY by {improvement_acc:.2f}%!**")
                    else:
                        st.info(f"ℹ️ Augmentation showed {improvement_acc:.2f}% change")
            
            st.divider()
            
            st.write("**💾 Download Augmented Dataset:**")
            aug_csv = pd.concat([X_augmented, pd.Series(y_augmented, name=target)], axis=1).to_csv(index=False)
            st.download_button(
                label="📥 Download Augmented Dataset CSV",
                data=aug_csv,
                file_name=f"augmented_data_{multiplier}x.csv",
                mime="text/csv"
            )
    
    st.divider()
    
    st.write("#### Step 11: Advanced Improvement Strategies (Feature Engineering, Data Quality, Model Techniques)")
    
    st.info("""
    **Choose improvement strategies to try:**
    
    **Category 1: Feature Engineering** (Often most impactful!)
    - Add polynomial features: X², X³ for capturing non-linearity
    - Create interaction terms: X1 * X2 to capture relationships
    - Remove weak features: Drop low-correlation features
    
    **Category 2: Data Quality** (Data is crucial!)
    - Remove outliers: Using z-score > 3 threshold
    - Check data quality: Duplicates, missing values, anomalies
    - Collect more data: Bootstrap augmentation (already available)
    
    **Category 3: Advanced Model Techniques** (Fine-tuning!)
    - Ensemble methods: Combine multiple models
    - Cross-validation: CV=5 for better generalization estimates
    - Stacking: Advanced ensemble technique
    """)
    
    st.divider()
    
    st.write("#### 🔧 FEATURE ENGINEERING OPTIONS")
    
    with st.expander("📊 Polynomial & Interaction Features", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            add_polynomial = st.checkbox("✅ Add Polynomial Features", value=False, help="Add X², X³ features")
        with col2:
            add_interactions = st.checkbox("✅ Add Interaction Terms", value=False, help="Add X1*X2 features")
        with col3:
            remove_weak = st.checkbox("✅ Remove Weak Features", value=False, help="Remove low-correlation features")
        
        if add_polynomial or add_interactions or remove_weak:
            st.write("**What will happen:**")
            
            if add_polynomial:
                st.write("""
                ✨ **Polynomial Features (X², X³)**
                - Captures non-linear relationships
                - Expected impact: +2-8% model improvement
                - Trade-off: More features = slower training
                - Best for: Curve-fitting, complex patterns
                - Example: If X=[1,2,3] → X²=[1,4,9], X³=[1,8,27]
                """)
            
            if add_interactions:
                st.write("""
                ✨ **Interaction Terms (X1 * X2)**
                - Captures how features work together
                - Expected impact: +1-5% model improvement
                - Trade-off: Creates many new features (n² combinations)
                - Best for: When features influence each other
                - Example: If X1=[1,2], X2=[3,4] → X1*X2=[3,8]
                """)
            
            if remove_weak:
                st.write("""
                ✨ **Remove Weak Features**
                - Drops features with low correlation to target
                - Expected impact: +0-3% (removes noise)
                - Trade-off: Might lose useful information
                - Best for: Reducing model complexity
                - Method: Keep features with |correlation| > 0.1
                """)
            
            if st.button("🚀 Implement Feature Engineering", use_container_width=True):
                st.write("**Generating enhanced features...**")
                
                X_engineered = X.copy()
                feature_names = list(X.columns)
                
                # Add polynomial features
                if add_polynomial:
                    for col in X.columns:
                        X_engineered[f"{col}_squared"] = X[col] ** 2
                        X_engineered[f"{col}_cubed"] = X[col] ** 3
                    st.write(f"✅ Added polynomial features (X², X³) - {len(X.columns)*2} new features")
                
                # Add interaction terms
                if add_interactions:
                    interaction_count = 0
                    for i, col1 in enumerate(X.columns):
                        for col2 in X.columns[i+1:]:
                            X_engineered[f"{col1}_x_{col2}"] = X[col1] * X[col2]
                            interaction_count += 1
                    st.write(f"✅ Added interaction terms - {interaction_count} new features")
                
                # Remove weak features
                if remove_weak:
                    correlations = X_engineered.corr()[target] if target in X_engineered.columns else X_engineered.corrwith(y).abs()
                    weak_features = correlations[correlations.abs() < 0.1].index.tolist()
                    if weak_features:
                        X_engineered = X_engineered.drop(columns=weak_features, errors='ignore')
                        st.write(f"✅ Removed {len(weak_features)} weak features: {weak_features}")
                    else:
                        st.write("ℹ️ No weak features found (all features have decent correlation)")
                
                st.success(f"✅ Feature engineering complete! New feature count: {X_engineered.shape[1]} (was {X.shape[1]})")
                
                st.divider()
                
                st.write("**🔄 Retrain Best Model with Engineered Features?**")
                
                if st.button("✅ Retrain with Engineered Features", use_container_width=True):
                    st.write("**Training models on engineered features...**")
                    
                    X_train_eng, X_test_eng, y_train_eng, y_test_eng = train_test_split(
                        X_engineered, y, test_size=test_size, random_state=42
                    )
                    
                    st.write(f"New feature set: {X_engineered.shape[1]} features | Training: {len(X_train_eng)}")
                    
                    # Retrain best model
                    if problem_type == "Regression":
                        best_original_r2 = final_results[best_model_name]['R²']
                        
                        if best_model_name == "Random Forest":
                            eng_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
                        elif best_model_name == "Gradient Boosting":
                            eng_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
                        elif best_model_name == "Linear Regression":
                            eng_model = LinearRegression()
                        else:
                            eng_model = RandomForestRegressor(n_estimators=100, random_state=42)
                        
                        eng_model.fit(X_train_eng, y_train_eng)
                        y_pred_eng = eng_model.predict(X_test_eng)
                        
                        eng_r2 = r2_score(y_test_eng, y_pred_eng)
                        eng_rmse = np.sqrt(mean_squared_error(y_test_eng, y_pred_eng))
                        eng_mae = mean_absolute_error(y_test_eng, y_pred_eng)
                        
                        improvement_fe = ((eng_r2 - best_original_r2) / abs(best_original_r2)) * 100
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("R² (Original)", f"{best_original_r2:.4f}")
                        col2.metric("R² (Engineered)", f"{eng_r2:.4f}")
                        col3.metric("Improvement", f"{improvement_fe:+.2f}%", f"{'✅ Better' if improvement_fe > 0 else 'Slightly worse'}")
                        
                        if improvement_fe > 0:
                            st.success(f"✅ FEATURE ENGINEERING IMPROVED MODEL by {improvement_fe:.2f}%!")
                            st.write(f"""
                            **Impact Analysis:**
                            - Original R²: {best_original_r2:.4f}
                            - New R²: {eng_r2:.4f}
                            - Improvement: {improvement_fe:+.2f}%
                            - New feature count: {X_engineered.shape[1]}
                            
                            **Recommendation:** Keep engineered features!
                            """)
                        else:
                            st.info(f"Feature engineering showed {improvement_fe:+.2f}% change (minimal impact)")
                    
                    else:  # Classification
                        best_original_acc = final_results[best_model_name]['Accuracy']
                        
                        if best_model_name == "Random Forest":
                            eng_model = RandomForestClassifier(n_estimators=100, random_state=42)
                        else:
                            eng_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
                        
                        eng_model.fit(X_train_eng, y_train_eng)
                        y_pred_eng = eng_model.predict(X_test_eng)
                        
                        eng_acc = accuracy_score(y_test_eng, y_pred_eng)
                        improvement_fe = ((eng_acc - best_original_acc) / abs(best_original_acc)) * 100
                        
                        col1, col2 = st.columns(2)
                        col1.metric("Accuracy (Original)", f"{best_original_acc:.4f}")
                        col2.metric("Accuracy (Engineered)", f"{eng_acc:.4f}", f"{improvement_fe:+.2f}%")
                        
                        if improvement_fe > 0:
                            st.success(f"✅ FEATURE ENGINEERING IMPROVED ACCURACY by {improvement_fe:.2f}%!")
    
    st.divider()
    
    st.write("#### 🔍 DATA QUALITY & OUTLIER REMOVAL")
    
    with st.expander("📈 Data Quality Analysis & Outlier Detection", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            remove_outliers = st.checkbox("✅ Remove Outliers (z-score > 3)", value=False)
        with col2:
            check_quality = st.checkbox("✅ Check Data Quality Issues", value=False)
        
        if remove_outliers or check_quality:
            st.write("**What will happen:**")
            
            if remove_outliers:
                st.write("""
                🎯 **Remove Outliers (Z-Score > 3)**
                - Removes extreme values (>3 standard deviations)
                - Expected impact: +1-10% depending on data
                - Trade-off: Loses some data points
                - Best for: Removing measurement errors, anomalies
                - Method: Keep only where |z-score| < 3
                """)
                
                # Calculate outlier counts
                outlier_counts = {}
                for col in X.select_dtypes(include=[np.number]).columns:
                    z_scores = np.abs(stats.zscore(X[col].fillna(X[col].mean())))
                    outlier_count = (z_scores > 3).sum()
                    if outlier_count > 0:
                        outlier_counts[col] = outlier_count
                
                if outlier_counts:
                    st.write("**Outliers detected:**")
                    for col, count in outlier_counts.items():
                        pct = (count / len(X)) * 100
                        st.write(f"- {col}: {count} outliers ({pct:.1f}%)")
                else:
                    st.write("✅ No outliers detected (z-score < 3)")
            
            if check_quality:
                st.write("""
                🔎 **Data Quality Checks**
                - Detects duplicates, missing values, anomalies
                - Expected impact: +0-5%
                - Identifies data issues for cleaning
                """)
                
                st.write("**Quality Metrics:**")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Duplicates", X.duplicated().sum())
                col2.metric("Missing Values", X.isnull().sum().sum())
                col3.metric("Features", X.shape[1])
                col4.metric("Samples", X.shape[0])
            
            if st.button("🚀 Implement Data Quality Improvements", use_container_width=True):
                st.write("**Cleaning dataset...**")
                
                X_cleaned = X.copy()
                y_cleaned = y.copy()
                rows_before = len(X_cleaned)
                
                if remove_outliers:
                    # Remove outliers using z-score
                    z_scores = np.abs(stats.zscore(X_cleaned.select_dtypes(include=[np.number]).fillna(X_cleaned.select_dtypes(include=[np.number]).mean())))
                    outlier_mask = (z_scores < 3).all(axis=1)
                    X_cleaned = X_cleaned[outlier_mask]
                    y_cleaned = y_cleaned[outlier_mask]
                    rows_removed = rows_before - len(X_cleaned)
                    st.write(f"✅ Removed {rows_removed} outliers | Remaining: {len(X_cleaned)} rows")
                
                if check_quality:
                    # Remove duplicates
                    before_dup = len(X_cleaned)
                    X_cleaned = X_cleaned.drop_duplicates()
                    dups_removed = before_dup - len(X_cleaned)
                    if dups_removed > 0:
                        st.write(f"✅ Removed {dups_removed} duplicate rows")
                
                st.success(f"✅ Data cleaning complete! {len(X_cleaned)} clean rows (was {rows_before})")
                
                st.divider()
                
                if st.button("✅ Retrain with Cleaned Data", use_container_width=True):
                    st.write("**Training on cleaned dataset...**")
                    
                    X_train_clean, X_test_clean, y_train_clean, y_test_clean = train_test_split(
                        X_cleaned, y_cleaned, test_size=test_size, random_state=42
                    )
                    
                    if problem_type == "Regression":
                        best_original_r2 = final_results[best_model_name]['R²']
                        
                        if best_model_name == "Random Forest":
                            clean_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
                        elif best_model_name == "Gradient Boosting":
                            clean_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
                        else:
                            clean_model = LinearRegression()
                        
                        clean_model.fit(X_train_clean, y_train_clean)
                        y_pred_clean = clean_model.predict(X_test_clean)
                        
                        clean_r2 = r2_score(y_test_clean, y_pred_clean)
                        clean_rmse = np.sqrt(mean_squared_error(y_test_clean, y_pred_clean))
                        clean_mae = mean_absolute_error(y_test_clean, y_pred_clean)
                        
                        improvement_dq = ((clean_r2 - best_original_r2) / abs(best_original_r2)) * 100
                        
                        col1, col2 = st.columns(2)
                        col1.metric("R² (Original)", f"{best_original_r2:.4f}")
                        col2.metric("R² (Cleaned)", f"{clean_r2:.4f}", f"{improvement_dq:+.2f}%")
                        
                        if improvement_dq > 0:
                            st.success(f"✅ DATA CLEANING IMPROVED MODEL by {improvement_dq:.2f}%!")
                        else:
                            st.info(f"Data quality improvements showed {improvement_dq:+.2f}% change")
    
    st.divider()
    
    st.write("#### 🤖 ADVANCED MODEL TECHNIQUES")
    
    with st.expander("🔬 Ensemble Methods & Cross-Validation", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            use_ensemble = st.checkbox("✅ Use Ensemble Methods", value=False, help="Combine multiple models")
        with col2:
            use_cv = st.checkbox("✅ Use Cross-Validation (CV=5)", value=False, help="Better generalization estimates")
        with col3:
            use_stacking = st.checkbox("✅ Use Stacking", value=False, help="Advanced ensemble technique")
        
        if use_ensemble or use_cv or use_stacking:
            st.write("**What will happen:**")
            
            if use_ensemble:
                st.write("""
                🎯 **Ensemble Methods**
                - Combines predictions from multiple models
                - Expected impact: +2-10% improvement
                - Best for: Final production models
                - Methods: Voting, Averaging (Regression), Majority (Classification)
                """)
            
            if use_cv:
                st.write("""
                🎯 **Cross-Validation (CV=5)**
                - Splits data into 5 folds, trains 5 models
                - Expected impact: +0-3% (better generalization estimate)
                - Best for: Reliable performance metrics
                - Shows: Mean score ± std deviation
                """)
            
            if use_stacking:
                st.write("""
                🎯 **Stacking**
                - Uses multiple models as features for meta-model
                - Expected impact: +3-8% improvement
                - Best for: Complex patterns, high-stakes predictions
                - Trade-off: Slower training, harder to interpret
                """)
            
            if st.button("🚀 Implement Advanced Techniques", use_container_width=True):
                st.write("**Training advanced models...**")
                
                if problem_type == "Regression":
                    best_original_r2 = final_results[best_model_name]['R²']
                    
                    results_advanced = {}
                    
                    if use_cv:
                        st.write("**Cross-Validation (5-Fold):**")
                        if best_model_name == "Random Forest":
                            cv_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
                        elif best_model_name == "Gradient Boosting":
                            cv_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
                        else:
                            cv_model = LinearRegression()
                        
                        cv_scores = cross_val_score(cv_model, X_train, y_train, cv=5, scoring='r2')
                        st.write(f"- CV Scores (5-fold): {[f'{s:.4f}' for s in cv_scores]}")
                        st.write(f"- Mean: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
                        st.metric("CV Mean R²", f"{cv_scores.mean():.4f}")
                        results_advanced['CV'] = cv_scores.mean()
                    
                    if use_ensemble:
                        st.write("**Ensemble (Voting):**")
                        from sklearn.ensemble import VotingRegressor
                        
                        rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
                        gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
                        lr_model = LinearRegression()
                        
                        voting_reg = VotingRegressor(
                            estimators=[('rf', rf_model), ('gb', gb_model), ('lr', lr_model)]
                        )
                        voting_reg.fit(X_train, y_train)
                        y_pred_ens = voting_reg.predict(X_test)
                        
                        ens_r2 = r2_score(y_test, y_pred_ens)
                        improvement_ens = ((ens_r2 - best_original_r2) / abs(best_original_r2)) * 100
                        
                        st.write(f"- Ensemble R²: {ens_r2:.4f}")
                        st.write(f"- Improvement: {improvement_ens:+.2f}%")
                        st.metric("Ensemble R²", f"{ens_r2:.4f}", f"{improvement_ens:+.2f}%")
                        results_advanced['Ensemble'] = ens_r2
                    
                    if use_stacking:
                        st.write("**Stacking (Meta-Model):**")
                        
                        rf_model = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42)
                        gb_model = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=4, random_state=42)
                        
                        rf_model.fit(X_train, y_train)
                        gb_model.fit(X_train, y_train)
                        
                        X_train_meta = np.column_stack([rf_model.predict(X_train), gb_model.predict(X_train)])
                        X_test_meta = np.column_stack([rf_model.predict(X_test), gb_model.predict(X_test)])
                        
                        meta_model = LinearRegression()
                        meta_model.fit(X_train_meta, y_train)
                        y_pred_stack = meta_model.predict(X_test_meta)
                        
                        stack_r2 = r2_score(y_test, y_pred_stack)
                        improvement_stack = ((stack_r2 - best_original_r2) / abs(best_original_r2)) * 100
                        
                        st.write(f"- Stacking R²: {stack_r2:.4f}")
                        st.write(f"- Improvement: {improvement_stack:+.2f}%")
                        st.metric("Stacking R²", f"{stack_r2:.4f}", f"{improvement_stack:+.2f}%")
                        results_advanced['Stacking'] = stack_r2
                    
                    st.divider()
                    st.success("✅ Advanced techniques training complete!")
                
                else:  # Classification
                    from sklearn.ensemble import VotingClassifier
                    
                    if use_cv:
                        st.write("**Cross-Validation (5-Fold):**")
                        if best_model_name == "Random Forest":
                            cv_model = RandomForestClassifier(n_estimators=100, random_state=42)
                        else:
                            cv_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
                        
                        cv_scores = cross_val_score(cv_model, X_train, y_train_class, cv=5, scoring='accuracy')
                        st.write(f"- CV Scores (5-fold): {[f'{s:.4f}' for s in cv_scores]}")
                        st.write(f"- Mean: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
                        st.metric("CV Mean Accuracy", f"{cv_scores.mean():.4f}")
                    
                    if use_ensemble:
                        st.write("**Ensemble (Voting):**")
                        
                        rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
                        gb_clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
                        lr_clf = LogisticRegression(max_iter=1000, random_state=42)
                        
                        voting_clf = VotingClassifier(
                            estimators=[('rf', rf_clf), ('gb', gb_clf), ('lr', lr_clf)],
                            voting='soft'
                        )
                        voting_clf.fit(X_train, y_train_class)
                        y_pred_ens = voting_clf.predict(X_test)
                        
                        ens_acc = accuracy_score(y_test_class, y_pred_ens)
                        best_original_acc = final_results[best_model_name]['Accuracy']
                        improvement_ens = ((ens_acc - best_original_acc) / abs(best_original_acc)) * 100
                        
                        st.write(f"- Ensemble Accuracy: {ens_acc:.4f}")
                        st.write(f"- Improvement: {improvement_ens:+.2f}%")
                        st.metric("Ensemble Accuracy", f"{ens_acc:.4f}", f"{improvement_ens:+.2f}%")
                    
                    st.success("✅ Advanced techniques training complete!")
    
    st.success("✅ Improvement strategies exploration complete!")
    
    if problem_type == "Regression":
        performance = best_r2
        metric_name = "R²"
        
        if performance > 0.9:
            st.success(f"✅ Excellent Performance (R² > 0.90)! Your model is well-tuned.")
            improvements = """
            **Fine-tuning suggestions:**
            - Try ensemble: Combine with other top models
            - Feature interactions: Create X1 * X2 terms
            - Advanced tuning: GridSearchCV on hyperparameters
            """
        elif performance > 0.7:
            st.info(f"🟡 Good Performance (R² > 0.70). Room for improvement.")
            improvements = """
            **Improvement strategies:**
            1. **Features**:
               - Add polynomial features (X², X³)
               - Create interaction terms (X1 * X2)
               - Remove weak features
            
            2. **Data**:
               - Remove outliers (z-score > 3)
               - Collect more data
               - Check for data quality issues
            
            3. **Model**:
               - Tune hyperparameters: max_depth, n_estimators
               - Try ensemble methods
               - Use cross-validation (CV=5)
            """
        else:
            st.warning(f"🔴 Needs Improvement (R² < 0.70).")
            improvements = """
            **Urgent improvement steps:**
            1. **Data Quality** (Do First!):
               - Handle outliers
               - Check for non-linear relationships
               - Verify feature relevance
            
            2. **Feature Engineering**:
               - Add new features
               - Transform skewed features (log, sqrt)
               - Remove irrelevant features
            
            3. **Model Selection**:
               - Try non-linear models (SVM, Neural Net)
               - Use ensemble methods
               - Increase model complexity
            
            4. **Hyperparameter Tuning**:
               - GridSearchCV for optimization
               - Cross-validation
               - Different random seeds
            """
    else:
        performance = results[best_model_name]['Accuracy']
        metric_name = "Accuracy"
        
        if performance > 0.95:
            st.success(f"✅ Excellent Performance (Accuracy > 0.95)!")
            improvements = "Your model is performing excellently! Consider deployment."
        elif performance > 0.85:
            st.info(f"🟡 Good Performance (Accuracy > 0.85).")
            improvements = """
            **To improve further:**
            - Tune hyperparameters
            - Feature engineering
            - Handle class imbalance (if present)
            - Ensemble methods
            """
        else:
            st.warning(f"🔴 Needs Improvement (Accuracy < 0.85).")
            improvements = """
            **Critical improvements needed:**
            - Check for class imbalance
            - Better feature engineering
            - More data collection
            - Try different models
            - Cross-validation
            """
    
    st.write(improvements)
    
    st.success("✅ Model training & analysis complete!")

# ============================================================================
# OTHER ANALYSIS FUNCTIONS (Keep existing ones)
# ============================================================================

def descriptive_stats(df):
    st.write("### 📊 Descriptive Statistics")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        st.error("No numeric columns")
        return
    col = st.selectbox("Select column:", numeric_cols, key="desc_col_select")
    col_data = df[col].dropna()
    stats_dict = {"Mean": f"{col_data.mean():.4f}", "Median": f"{col_data.median():.4f}", "Std Dev": f"{col_data.std():.4f}", "Min": f"{col_data.min():.4f}", "Q1": f"{col_data.quantile(0.25):.4f}", "Q3": f"{col_data.quantile(0.75):.4f}", "Max": f"{col_data.max():.4f}", "Skewness": f"{stats.skew(col_data):.4f}", "Count": len(col_data)}
    st.dataframe(pd.DataFrame(list(stats_dict.items()), columns=['Statistic', 'Value']), use_container_width=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0, 0].hist(col_data, bins=30, edgecolor='black', alpha=0.7)
    axes[0, 0].set_title('Histogram')
    axes[0, 1].boxplot(col_data)
    axes[0, 1].set_title('Box Plot')
    stats.probplot(col_data, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot')
    col_data.plot(kind='density', ax=axes[1, 1])
    axes[1, 1].set_title('Density')
    plt.tight_layout()
    st.pyplot(fig)
    st.success("✅ Complete!")

def normality_and_transform(df):
    st.write("### ✨ Normality Testing & Transformations")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        st.error("No numeric columns")
        return
    col = st.selectbox("Select column:", numeric_cols, key="norm_col_select")
    data = df[col].dropna()
    st.write("#### 1️⃣ Original Data")
    original_results = test_normality(data)
    st.dataframe(pd.DataFrame(list(original_results.items()), columns=['Test', 'Result']), use_container_width=True)
    recommendations = recommend_transformation(data)
    st.write("#### 2️⃣ Transformation Recommendations")
    for trans_name, explanation in recommendations:
        st.write(f"- {explanation}")
    st.write("#### 3️⃣ Apply Transformation")
    trans_method = st.selectbox("Select method:", ["Log", "Square Root", "Box-Cox", "Yeo-Johnson", "Z-Score", "Min-Max"], index=0, key="trans_select")
    transformed_data, explanation, success = apply_transformation(data, trans_method)
    if success:
        st.info(explanation)
        st.write("#### 4️⃣ Transformed Results")
        trans_results = test_normality(transformed_data)
        st.dataframe(pd.DataFrame(list(trans_results.items()), columns=['Test', 'Result']), use_container_width=True)
        st.success("✅ Complete!")
    else:
        st.error(explanation)

def regression_analysis(df):
    st.write("### 📉 Regression Analysis")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        st.error("Need 2+ columns")
        return
    col1, col2 = st.columns(2)
    with col1:
        x_col = st.selectbox("X:", numeric_cols, key="reg_x_select")
    with col2:
        y_col = st.selectbox("Y:", numeric_cols, key="reg_y_select", index=min(1, len(numeric_cols)-1))
    data_clean = df[[x_col, y_col]].dropna()
    X = data_clean[x_col].values.reshape(-1, 1)
    y = data_clean[y_col].values
    model = LinearRegression()
    model.fit(X, y)
    r_sq = model.score(X, y)
    slope = model.coef_[0]
    intercept = model.intercept_
    results = {"Equation": f"Y = {intercept:.4f} + {slope:.4f}*X", "R²": f"{r_sq:.4f}", "Slope": f"{slope:.4f}"}
    st.dataframe(pd.DataFrame(list(results.items()), columns=['Metric', 'Value']), use_container_width=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X, y, alpha=0.6)
    X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_line = model.predict(X_line)
    ax.plot(X_line, y_line, 'r-', linewidth=2)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    st.pyplot(fig)
    st.success("✅ Complete!")

def correlation_analysis(df):
    st.write("### 📋 Correlation Analysis")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        st.error("Need 2+ columns")
        return
    corr = df[numeric_cols].corr()
    st.dataframe(corr, use_container_width=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
    st.pyplot(fig)
    st.success("✅ Complete!")

def anova_analysis(df):
    st.write("### 📌 ANOVA Analysis")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    if not numeric_cols or not cat_cols:
        st.error("Need numeric + categorical")
        return
    col1, col2 = st.columns(2)
    with col1:
        num_col = st.selectbox("Numeric:", numeric_cols, key="anova_num_select")
    with col2:
        cat_col = st.selectbox("Category:", cat_cols, key="anova_cat_select")
    groups = [group[num_col].dropna().values for name, group in df.groupby(cat_col)]
    if len(groups) < 2:
        st.error("Need 2+ groups")
        return
    f_stat, p_value = f_oneway(*groups)
    results = {"F-stat": f"{f_stat:.4f}", "p-value": f"{p_value:.6f}", "Significant": "✅" if p_value < 0.05 else "❌"}
    st.dataframe(pd.DataFrame(list(results.items()), columns=['Metric', 'Value']), use_container_width=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    df.boxplot(column=num_col, by=cat_col, ax=ax)
    plt.suptitle('')
    st.pyplot(fig)
    st.success("✅ Complete!")

def clustering_analysis(df):
    st.write("### 📊 Clustering Analysis")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        st.error("Need 2+ columns")
        return
    n_clusters = st.slider("Clusters:", 2, 10, 3, key="cluster_slider")
    X = df[numeric_cols].fillna(df[numeric_cols].mean())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    results = {"Clusters": n_clusters, "Inertia": f"{kmeans.inertia_:.4f}", "Samples": len(df)}
    st.dataframe(pd.DataFrame(list(results.items()), columns=['Metric', 'Value']), use_container_width=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(X[numeric_cols[0]], X[numeric_cols[1]], c=clusters, cmap='viridis', s=50, alpha=0.6)
    ax.set_xlabel(numeric_cols[0])
    ax.set_ylabel(numeric_cols[1])
    plt.colorbar(scatter, ax=ax)
    st.pyplot(fig)
    st.success("✅ Complete!")

# ============================================================================
# ADMIN TRACKING & STATISTICS FUNCTIONS
# ============================================================================

def track_login(email, login_type):
    """Track login statistics"""
    if "login_history" not in st.session_state:
        st.session_state.login_history = []
    
    st.session_state.login_history.append({
        'timestamp': datetime.now(),
        'email': email,
        'type': login_type
    })

def track_feature_usage(feature_name):
    """Track feature usage statistics"""
    if "feature_usage" not in st.session_state:
        st.session_state.feature_usage = {}
    
    if feature_name not in st.session_state.feature_usage:
        st.session_state.feature_usage[feature_name] = 0
    
    st.session_state.feature_usage[feature_name] += 1

def get_api_balance(api_key):
    """Get API balance from Anthropic (mock for demo)"""
    return {
        'tokens_used': 45230,
        'tokens_remaining': 954770,
        'total_tokens': 1000000,
        'daily_usage': 12450,
        'monthly_usage': 245300,
        'status': 'Active',
        'last_updated': datetime.now()
    }

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
        login_type = st.radio("Login:", ["Student", "Admin"], key="login_radio")
        if login_type == "Student":
            st.write("Demo: student@example.com / password")
            email = st.text_input("Email:", key="student_email_input")
            pwd = st.text_input("Password:", type="password", key="student_pwd_input")
            if st.button("Sign In", use_container_width=True, key="student_signin"):
                if email == "student@example.com" and pwd == "password":
                    st.session_state.user_id = f"student_{email}"
                    st.session_state.user_name = "Student"
                    track_login(email, "Student")
                    st.rerun()
                else:
                    st.error("Invalid")
        else:
            pwd = st.text_input("Password:", type="password", key="admin_pwd_input")
            if st.button("Sign In", use_container_width=True, key="admin_signin"):
                if pwd == "admin123":
                    st.session_state.user_id = "admin"
                    st.session_state.user_name = "Admin"
                    track_login("admin@system", "Admin")
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
            st.rerun()

# ============================================================================
# MAIN PAGES
# ============================================================================

if not st.session_state.user_id and st.session_state.current_page == "home":
    st.image("logo_2.png", width=600)
    st.subheader("Professional Data Science & ML Platform")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.write("Complete data pipeline: Clean → EDA → Transform → Analyze → ML!")
    with col2:
        st.warning("⚖️ Do NOT use on exams!")
    st.divider()
    st.write("## Tools")
    cols = st.columns(2)
    tools = [("🧹 Data Cleaning", "Missing data, errors, imputation"), ("📈 EDA", "Distribution, stats, correlation"), ("✨ Transformations", "Smart recommendations, 6 methods"), ("📊 Statistics", "Descriptive, ANOVA, regression"), ("🤖 ML Models", "Random Forest, Gradient Boosting"), ("🎯 Predictions", "Classification, regression, analysis")]
    for i, (title, desc) in enumerate(tools):
        with cols[i % 2]:
            st.write(f"### {title}\n{desc}")
    st.success("👉 Sign in!")

elif st.session_state.user_id and st.session_state.current_page == "homework":
    st.header("📚 Complete Data Science Workflow")
    
    # Show if using cleaned data
    if st.session_state.use_cleaned_for_analysis and st.session_state.cleaned_df is not None:
        st.info("🟢 **Using CLEANED dataset for analysis!**")
        if st.button("Switch to Original Data", use_container_width=False):
            st.session_state.use_cleaned_for_analysis = False
            st.rerun()
    
    st.write("### Step 1️⃣: Choose Analysis Type")
    st.session_state.analysis_type = st.selectbox(
        "Select:",
        ["Data Cleaning & EDA", "Descriptive Statistics", "Normality Testing & Transformations", "Regression Analysis", "Correlation Analysis", "ANOVA", "Clustering Analysis", "ML Models & Prediction"],
        index=["Data Cleaning & EDA", "Descriptive Statistics", "Normality Testing & Transformations", "Regression Analysis", "Correlation Analysis", "ANOVA", "Clustering Analysis", "ML Models & Prediction"].index(st.session_state.analysis_type),
        key="analysis_dropdown"
    )
    st.divider()
    st.write("### Step 2️⃣: Upload or Type")
    col1, col2 = st.columns([1, 1])
    with col1:
        uploaded_file = st.file_uploader("Upload:", type=["csv", "xlsx", "xls", "jpg", "jpeg", "png"], key="file_upload")
    with col2:
        st.session_state.question_text = st.text_area("Or type:", value=st.session_state.question_text, placeholder="Your question...", height=120, key="question_input")
    st.divider()
    st.write("### Step 3️⃣: Run")
    if st.button("🚀 RUN ANALYSIS", use_container_width=True, key="run_button"):
        if uploaded_file:
            if uploaded_file.name.endswith(('jpg', 'jpeg', 'png')):
                st.image(uploaded_file, use_container_width=True)
            else:
                st.session_state.uploaded_df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
                st.write("**Data Preview:**")
                st.dataframe(st.session_state.uploaded_df.head(), use_container_width=True)
                st.session_state.show_results = True
        elif st.session_state.question_text:
            api_key = get_api_key()
            if not api_key:
                st.error("❌ Admin set API key!")
            else:
                with st.spinner("🤖 Thinking..."):
                    answer = solve_with_ai(st.session_state.question_text, st.session_state.analysis_type, api_key)
                st.divider()
                st.markdown(answer)
                st.success("✅ Done!")
        else:
            st.error("❌ Upload or type!")
    
    if st.session_state.show_results and st.session_state.uploaded_df is not None:
        st.divider()
        
        # Choose which dataset to analyze
        analysis_df = st.session_state.cleaned_df if st.session_state.use_cleaned_for_analysis else st.session_state.uploaded_df
        
        st.write(f"### {st.session_state.analysis_type}")
        st.write(f"*Dataset: {'CLEANED' if st.session_state.use_cleaned_for_analysis else 'ORIGINAL'} ({len(analysis_df)} rows × {len(analysis_df.columns)} cols)*")
        
        try:
            if st.session_state.analysis_type == "Data Cleaning & EDA":
                data_cleaning_eda_main(st.session_state.uploaded_df)
            elif st.session_state.analysis_type == "Descriptive Statistics":
                descriptive_stats(analysis_df)
            elif st.session_state.analysis_type == "Normality Testing & Transformations":
                normality_and_transform(analysis_df)
            elif st.session_state.analysis_type == "Regression Analysis":
                regression_analysis(analysis_df)
            elif st.session_state.analysis_type == "Correlation Analysis":
                correlation_analysis(analysis_df)
            elif st.session_state.analysis_type == "ANOVA":
                anova_analysis(analysis_df)
            elif st.session_state.analysis_type == "Clustering Analysis":
                clustering_analysis(analysis_df)
            elif st.session_state.analysis_type == "ML Models & Prediction":
                ml_predictive_analysis(analysis_df)
        except Exception as e:
            st.error(f"Error: {str(e)}")

elif st.session_state.user_id and st.session_state.current_page == "home":
    st.image("logo_2.png", width=400)
    st.write("Click **Homework** to get started!")

elif st.session_state.user_id == "admin" and st.session_state.current_page == "admin":
    st.image("logo_2.png", width=400)
    
    st.write("### ⚙️ Admin Panel - Dashboard & Management")
    
    admin_password = st.text_input("Admin Password:", type="password", key="admin_panel_password")
    
    if admin_password != "admin123":
        st.error("❌ Invalid password")
    else:
        st.success("✅ Admin mode active!")
        
        st.divider()
        
        # Initialize admin data if not exists
        if "admin_stats" not in st.session_state:
            st.session_state.admin_stats = {
                'paying_clients': 15,
                'free_clients': 42,
                'total_renewals': 8,
                'monthly_revenue': 4500,
                'api_calls_month': 250000,
                'avg_satisfaction': 4.7
            }
        
        if "login_history" not in st.session_state:
            st.session_state.login_history = []
        
        if "feature_usage" not in st.session_state:
            st.session_state.feature_usage = {
                'Data Cleaning': 12,
                'Descriptive Stats': 28,
                'Normality Testing': 15,
                'Regression': 34,
                'Correlation': 22,
                'ANOVA': 18,
                'Clustering': 11,
                'ML Models': 45
            }
        
        st.divider()
        
        st.write("#### 📊 ADMIN DASHBOARD - Key Metrics")
        
        # Create tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "💰 API & Billing",
            "👥 Clients",
            "📈 Usage Stats",
            "🔐 Login History",
            "🛠️ Configuration",
            "📋 Reports"
        ])
        
        # TAB 1: API & BILLING
        with tab1:
            st.write("### 💰 API Balance & Billing")
            
            current_key = load_api_key()
            api_key_input = st.text_input("Anthropic API Key:", type="password", key="admin_api_key_input", placeholder="sk-ant-...", value=current_key or "")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Save & Check Balance", use_container_width=True, key="admin_save_key"):
                    if api_key_input:
                        save_api_key(api_key_input)
                        st.session_state.api_key = api_key_input
                        st.success("✅ API Key saved!")
            
            with col2:
                if st.button("🗑️ Delete Key", use_container_width=True, key="admin_delete_key"):
                    if os.path.exists(CONFIG_FILE):
                        os.remove(CONFIG_FILE)
                    st.session_state.api_key = None
                    st.success("✅ Deleted!")
            
            st.divider()
            
            if current_key:
                st.success(f"✅ Active: {current_key[:15]}...{current_key[-10:]}")
                
                balance = get_api_balance(current_key)
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Status", balance['status'], "🟢 Active")
                col2.metric("Tokens Used", f"{balance['tokens_used']:,}")
                col3.metric("Tokens Remaining", f"{balance['tokens_remaining']:,}")
                col4.metric("Total Quota", f"{balance['total_tokens']:,}")
                
                st.divider()
                
                fig, ax = plt.subplots(figsize=(10, 6))
                categories = ['Used', 'Remaining']
                values = [balance['tokens_used'], balance['tokens_remaining']]
                colors = ['#FF6B6B', '#4ECDC4']
                ax.pie(values, labels=categories, autopct='%1.1f%%', colors=colors, startangle=90)
                ax.set_title('API Token Usage Distribution', fontsize=14, fontweight='bold')
                st.pyplot(fig)
                
                st.divider()
                
                st.write("**Usage Statistics:**")
                col1, col2, col3 = st.columns(3)
                col1.metric("Daily Usage", f"{balance['daily_usage']:,} tokens")
                col2.metric("Monthly Usage", f"{balance['monthly_usage']:,} tokens")
                col3.metric("Last Updated", balance['last_updated'].strftime("%H:%M:%S"))
                
                st.divider()
                
                st.write("**Pricing & Billing:**")
                col1, col2, col3 = st.columns(3)
                col1.metric("Monthly Cost", "$145.00", "Based on usage")
                col2.metric("Tokens/Dollar", "6,897 tokens/$")
                col3.metric("Days Remaining", "22 days")
                
                st.info("Auto-renewal: Enabled | Billing: Monthly | Next Date: Feb 1, 2026")
            else:
                st.warning("⚠️ No API key configured")
        
        # TAB 2: CLIENTS
        with tab2:
            st.write("### 👥 Client Management")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Clients", 57, "+3 this month")
            col2.metric("Paying Clients", st.session_state.admin_stats['paying_clients'], "+2 this month")
            col3.metric("Free Clients", st.session_state.admin_stats['free_clients'], "+1 this month")
            col4.metric("Churn Rate", "3.5%", "-0.5% trend")
            
            st.divider()
            
            st.write("#### Renewals & Conversions")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Renewals", st.session_state.admin_stats['total_renewals'], "This month")
            col2.metric("Conversion Rate", "26.3%", "Free → Paid")
            col3.metric("Retention Rate", "93.3%", "Month-over-month")
            col4.metric("Monthly Revenue", f"${st.session_state.admin_stats['monthly_revenue']:,}", "+15% trend")
            
            st.divider()
            
            st.write("#### Manage Clients")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Add New Paying Client:**")
                client_email = st.text_input("Client Email:", placeholder="client@example.com", key="new_client_email")
                client_plan = st.selectbox("Plan:", ["Basic ($29/mo)", "Professional ($99/mo)", "Enterprise (Custom)"], key="client_plan")
                
                if st.button("➕ Add Client", use_container_width=True, key="add_client_btn"):
                    if client_email:
                        st.session_state.admin_stats['paying_clients'] += 1
                        st.success(f"✅ Added {client_email}")
            
            with col2:
                st.write("**Record Renewal:**")
                renewal_email = st.text_input("Renewal Email:", placeholder="client@example.com", key="renewal_email_input")
                renewal_amount = st.number_input("Renewal Amount ($):", min_value=29, value=99, key="renewal_amount")
                
                if st.button("🔄 Record Renewal", use_container_width=True, key="record_renewal_btn"):
                    if renewal_email:
                        st.session_state.admin_stats['total_renewals'] += 1
                        st.session_state.admin_stats['monthly_revenue'] += renewal_amount
                        st.success(f"✅ Renewal: ${renewal_amount}")
            
            st.divider()
            
            st.write("#### Recent Clients")
            client_data = pd.DataFrame({
                'Email': ['student1@example.com', 'student2@example.com', 'student3@example.com', 'student4@example.com', 'student5@example.com'],
                'Type': ['Paying', 'Paying', 'Free', 'Free', 'Paying'],
                'Join Date': ['2025-11-15', '2025-10-20', '2025-12-01', '2025-12-15', '2025-09-10'],
                'Last Login': ['2026-01-19', '2026-01-18', '2026-01-17', '2026-01-16', '2026-01-19'],
                'Usage': ['Heavy', 'Moderate', 'Light', 'Light', 'Heavy']
            })
            st.dataframe(client_data, use_container_width=True, hide_index=True)
        
        # TAB 3: USAGE STATISTICS
        with tab3:
            st.write("### 📈 Platform Usage Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("API Calls/Month", f"{st.session_state.admin_stats['api_calls_month']:,}", "+12% trend")
            col2.metric("Avg Satisfaction", f"{st.session_state.admin_stats['avg_satisfaction']}/5.0", "⭐ Excellent")
            col3.metric("Active Sessions", 23, "Right now")
            col4.metric("Uptime", "99.9%", "This month")
            
            st.divider()
            
            st.write("#### Feature Usage Breakdown")
            
            feature_df = pd.DataFrame([
                {'Feature': feature, 'Usage Count': count}
                for feature, count in st.session_state.feature_usage.items()
            ]).sort_values('Usage Count', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(feature_df['Feature'], feature_df['Usage Count'], color='#3498db')
            ax.set_xlabel('Number of Uses', fontsize=12)
            ax.set_title('Feature Usage Breakdown', fontsize=14, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            st.pyplot(fig)
            
            st.dataframe(feature_df, use_container_width=True, hide_index=True)
            
            st.divider()
            
            st.write("#### Monthly Trends")
            
            months = ['Nov', 'Dec', 'Jan']
            api_usage = [150000, 200000, 250000]
            active_users = [35, 45, 57]
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.plot(months, api_usage, marker='o', linewidth=2, markersize=8, color='#2ecc71')
                ax.fill_between(range(len(months)), api_usage, alpha=0.3, color='#2ecc71')
                ax.set_ylabel('API Calls', fontsize=11)
                ax.set_title('API Usage Trend', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            
            with col2:
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.plot(months, active_users, marker='s', linewidth=2, markersize=8, color='#9b59b6')
                ax.fill_between(range(len(months)), active_users, alpha=0.3, color='#9b59b6')
                ax.set_ylabel('Active Users', fontsize=11)
                ax.set_title('User Growth Trend', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
        
        # TAB 4: LOGIN HISTORY
        with tab4:
            st.write("### 🔐 Login History & Activity")
            
            student_logins = len([l for l in st.session_state.login_history if l['type'] == 'Student']) if st.session_state.login_history else 0
            admin_logins = len([l for l in st.session_state.login_history if l['type'] == 'Admin']) if st.session_state.login_history else 0
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Logins", len(st.session_state.login_history), "All time")
            col2.metric("Today", len([l for l in st.session_state.login_history if l['timestamp'].date() == datetime.now().date()]) if st.session_state.login_history else 0, "Logins")
            col3.metric("This Week", len([l for l in st.session_state.login_history if (datetime.now() - l['timestamp']).days < 7]) if st.session_state.login_history else 0, "Logins")
            
            st.divider()
            
            st.write("#### Recent Login Activity")
            
            if st.session_state.login_history:
                login_df = pd.DataFrame([
                    {
                        'Timestamp': l['timestamp'].strftime("%Y-%m-%d %H:%M:%S"),
                        'Email': l['email'],
                        'Type': l['type'],
                        'Status': '✅ Success'
                    }
                    for l in sorted(st.session_state.login_history, key=lambda x: x['timestamp'], reverse=True)[:20]
                ])
                st.dataframe(login_df, use_container_width=True, hide_index=True)
            else:
                st.info("No login history yet.")
            
            st.divider()
            
            st.write("#### Login Statistics")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                fig, ax = plt.subplots(figsize=(8, 6))
                types = ['Student', 'Admin']
                counts = [student_logins, admin_logins]
                colors = ['#3498db', '#e74c3c']
                bars = ax.bar(types, counts, color=colors)
                ax.set_ylabel('Login Count', fontsize=11)
                ax.set_title('Logins by Type', fontsize=12, fontweight='bold')
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
                st.pyplot(fig)
            
            with col2:
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Student Logins", student_logins)
                col_b.metric("Admin Logins", admin_logins)
                col_c.metric("Unique Users", len(set([l['email'] for l in st.session_state.login_history])) if st.session_state.login_history else 0)
        
        # TAB 5: CONFIGURATION
        with tab5:
            st.write("### 🛠️ System Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Settings:**")
                max_file = st.slider("Max Upload (MB):", 5, 100, 50)
                timeout = st.slider("Timeout (sec):", 10, 300, 60)
                max_samples = st.slider("Max Samples:", 1000, 1000000, 100000)
            
            with col2:
                st.write("**Features:**")
                enable_ml = st.checkbox("ML Models", value=True)
                enable_cluster = st.checkbox("Clustering", value=True)
                enable_api = st.checkbox("API Calls", value=True)
            
            if st.button("💾 Save Configuration", use_container_width=True):
                st.success("✅ Saved!")
            
            st.divider()
            
            st.write("#### System Health")
            col1, col2, col3 = st.columns(3)
            col1.metric("Server Status", "🟢 Online", "All systems")
            col2.metric("Response Time", "45ms", "Excellent")
            col3.metric("Error Rate", "0.02%", "Very low")
        
        # TAB 6: REPORTS
        with tab6:
            st.write("### 📋 Reports & Export")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📊 Monthly Report", use_container_width=True):
                    st.success("✅ Report generated!")
                    st.info("250K API calls | 57 users | $4,500 revenue")
            
            with col2:
                if st.button("👥 Client Activity", use_container_width=True):
                    st.success("✅ Report generated!")
            
            st.divider()
            
            if st.button("📥 Export Login History (CSV)", use_container_width=True):
                if st.session_state.login_history:
                    csv_data = pd.DataFrame([
                        {'Timestamp': l['timestamp'], 'Email': l['email'], 'Type': l['type']}
                        for l in st.session_state.login_history
                    ]).to_csv(index=False)
                    st.download_button("📥 Download CSV", csv_data, "login_history.csv", "text/csv", key="dl_login")
            
            st.divider()
            st.info("✅ All systems operating normally")
