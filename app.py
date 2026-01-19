import streamlit as st
import requests
import os
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import ttest_ind, f_oneway, shapiro, anderson, kstest, boxcox, yeojohnson
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
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

def ml_predictive_analysis(df):
    """Advanced ML models for predictive analysis"""
    st.write("### 🤖 ML Models & Predictive Analysis")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.error("Need at least 2 numeric columns")
        return
    
    st.write("#### Step 1: Choose Target Variable")
    target = st.selectbox("Select target (Y):", numeric_cols, key="ml_target")
    
    feature_cols = [col for col in numeric_cols if col != target]
    
    if not feature_cols:
        st.error("Need at least 1 feature besides target")
        return
    
    st.write("#### Step 2: Select Features")
    selected_features = st.multiselect("Select features (X):", feature_cols, default=feature_cols, key="ml_features")
    
    if not selected_features:
        st.error("Select at least 1 feature")
        return
    
    st.write("#### Step 3: Choose Model Type")
    model_type = st.selectbox("Model Type:", ["Regression", "Classification"], key="ml_type")
    
    st.write("#### Step 4: Train Test Split")
    test_size = st.slider("Test size:", 0.1, 0.5, 0.2, key="ml_test_size")
    
    X = df[selected_features].fillna(df[selected_features].mean())
    y = df[target].fillna(df[target].mean())
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    st.write(f"Training: {len(X_train)} | Testing: {len(X_test)}")
    
    st.divider()
    
    st.write("#### Step 5: Train Models")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Random Forest**")
        if model_type == "Regression":
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
            rf_model.fit(X_train, y_train)
            rf_pred = rf_model.predict(X_test)
            rf_r2 = r2_score(y_test, rf_pred)
            rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
            rf_mae = mean_absolute_error(y_test, rf_pred)
            results['Random Forest'] = {'R²': rf_r2, 'RMSE': rf_rmse, 'MAE': rf_mae, 'model': rf_model, 'pred': rf_pred}
            st.write(f"R²: {rf_r2:.4f} | RMSE: {rf_rmse:.4f} | MAE: {rf_mae:.4f}")
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(y_test, rf_pred, alpha=0.6)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            ax.set_xlabel('Actual')
            ax.set_ylabel('Predicted')
            ax.set_title('Random Forest Predictions')
            st.pyplot(fig)
        else:
            y_class = (y > y.median()).astype(int)
            y_train_class = (y_train > y.median()).astype(int)
            y_test_class = (y_test > y.median()).astype(int)
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train_class)
            rf_pred = rf_model.predict(X_test)
            rf_acc = accuracy_score(y_test_class, rf_pred)
            results['Random Forest'] = {'Accuracy': rf_acc, 'model': rf_model, 'pred': rf_pred}
            st.write(f"Accuracy: {rf_acc:.4f}")
    
    with col2:
        st.write("**Gradient Boosting**")
        if model_type == "Regression":
            gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5)
            gb_model.fit(X_train, y_train)
            gb_pred = gb_model.predict(X_test)
            gb_r2 = r2_score(y_test, gb_pred)
            gb_rmse = np.sqrt(mean_squared_error(y_test, gb_pred))
            gb_mae = mean_absolute_error(y_test, gb_pred)
            results['Gradient Boosting'] = {'R²': gb_r2, 'RMSE': gb_rmse, 'MAE': gb_mae, 'model': gb_model, 'pred': gb_pred}
            st.write(f"R²: {gb_r2:.4f} | RMSE: {gb_rmse:.4f} | MAE: {gb_mae:.4f}")
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(y_test, gb_pred, alpha=0.6, color='orange')
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            ax.set_xlabel('Actual')
            ax.set_ylabel('Predicted')
            ax.set_title('Gradient Boosting Predictions')
            st.pyplot(fig)
        else:
            y_class = (y > y.median()).astype(int)
            y_train_class = (y_train > y.median()).astype(int)
            y_test_class = (y_test > y.median()).astype(int)
            gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
            gb_model.fit(X_train, y_train_class)
            gb_pred = gb_model.predict(X_test)
            gb_acc = accuracy_score(y_test_class, gb_pred)
            results['Gradient Boosting'] = {'Accuracy': gb_acc, 'model': gb_model, 'pred': gb_pred}
            st.write(f"Accuracy: {gb_acc:.4f}")
    
    st.divider()
    
    st.write("#### Step 6: Model Comparison")
    comp_data = []
    for name, metrics in results.items():
        comp_data.append({**{'Model': name}, **{k: f"{v:.4f}" if isinstance(v, float) else v for k, v in metrics.items() if k not in ['model', 'pred']}})
    st.dataframe(pd.DataFrame(comp_data), use_container_width=True)
    
    st.divider()
    
    st.write("#### Step 7: Model Improvement Guide")
    st.write("""
    **📈 How to Improve Your Model:**
    
    **1️⃣ Performance Issues**
    - Low R² / Accuracy?
      - ✅ Add more features
      - ✅ Remove irrelevant features
      - ✅ Collect more data
      - ✅ Try different hyperparameters
    
    **2️⃣ Overfitting (Train >> Test)**
    - ✅ Reduce max_depth
    - ✅ Increase min_samples_split
    - ✅ Add regularization (L1/L2)
    - ✅ Get more training data
    - ✅ Feature engineering
    
    **3️⃣ Underfitting (Train ≈ Test, both low)**
    - ✅ Increase model complexity
    - ✅ Increase max_depth
    - ✅ Add polynomial features
    - ✅ Use better features
    - ✅ Reduce regularization
    
    **4️⃣ Data Issues**
    - ✅ Check for missing values (impute)
    - ✅ Remove outliers (z-score > 3)
    - ✅ Scale features (StandardScaler)
    - ✅ Handle imbalanced classes (SMOTE)
    - ✅ Feature normalization
    
    **5️⃣ Feature Engineering**
    - ✅ Create interaction terms (X1 * X2)
    - ✅ Polynomial features (X^2, X^3)
    - ✅ Log transform skewed features
    - ✅ Domain-specific features
    - ✅ PCA for dimensionality reduction
    
    **6️⃣ Hyperparameter Tuning**
    - ✅ GridSearchCV for best params
    - ✅ Adjust learning_rate (GB)
    - ✅ Adjust n_estimators
    - ✅ Adjust max_depth
    - ✅ Cross-validation (CV=5)
    
    **7️⃣ Model Selection**
    - For Linear: LinearRegression, Ridge, Lasso
    - For Complex: Random Forest, GB, XGBoost
    - For Classification: LogisticRegression, RFC, GBC
    - Ensemble: VotingRegressor, StackingRegressor
    """)
    
    st.success("✅ Model training complete!")

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
                    st.rerun()
                else:
                    st.error("Invalid")
        else:
            pwd = st.text_input("Password:", type="password", key="admin_pwd_input")
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
            st.rerun()

# ============================================================================
# MAIN PAGES
# ============================================================================

if not st.session_state.user_id and st.session_state.current_page == "home":
    st.title("📊 Data Insight Studio")
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
    st.title("📊 Data Insight Studio")
    st.write("Click Homework!")

elif st.session_state.user_id == "admin" and st.session_state.current_page == "admin":
    st.title("⚙️ Admin Panel")
    st.divider()
    current_key = load_api_key()
    if current_key:
        st.success(f"✅ Active: {current_key[:20]}...")
    else:
        st.warning("⚠️ No key")
    st.divider()
    api_key_input = st.text_input("API Key:", type="password", value=current_key or "", key="api_key_input")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Save", use_container_width=True, key="save_key_btn"):
            if api_key_input:
                if save_api_key(api_key_input):
                    st.session_state.api_key = api_key_input
                    st.success("✅ Saved!")
            else:
                st.error("Enter key")
    with col2:
        if st.button("🗑️ Delete", use_container_width=True, key="delete_key_btn"):
            try:
                if os.path.exists(CONFIG_FILE):
                    os.remove(CONFIG_FILE)
                st.session_state.api_key = None
                st.success("✅ Deleted!")
            except:
                st.error("Error")
