import streamlit as st
import requests
import base64
import json
import os
from datetime import datetime
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import io

# ============================================================================
# CONFIGURATION
# ============================================================================

st.set_page_config(page_title="Data Insight Studio", layout="wide")

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

# ============================================================================
# PERSISTENT API KEY STORAGE
# ============================================================================

def save_api_key(key):
    """Save API key to file"""
    try:
        with open(CONFIG_FILE, "w") as f:
            f.write(key)
        return True
    except:
        return False

def load_api_key():
    """Load API key from file or environment variable"""
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
    """Get API key from session or file"""
    if st.session_state.api_key:
        return st.session_state.api_key
    
    file_key = load_api_key()
    if file_key:
        st.session_state.api_key = file_key
        return file_key
    
    return None

# ============================================================================
# HELPER: CONVERT IMAGE TO BASE64
# ============================================================================

def image_to_base64(image_file):
    return base64.b64encode(image_file.read()).decode()

# ============================================================================
# HELPER: READ EXCEL/CSV FILES
# ============================================================================

def read_excel_csv(file):
    """Read Excel or CSV file and convert to dataframe"""
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

# ============================================================================
# STATISTICAL ANALYSIS FUNCTIONS
# ============================================================================

def calculate_descriptive_stats(df, column):
    """Calculate descriptive statistics"""
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
    """Create standard statistical visualizations"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Histogram
    axes[0, 0].hist(df[column].dropna(), bins=30, edgecolor='black', alpha=0.7)
    axes[0, 0].set_title(f'Histogram of {column}')
    axes[0, 0].set_xlabel('Value')
    axes[0, 0].set_ylabel('Frequency')
    
    # Box plot
    axes[0, 1].boxplot(df[column].dropna())
    axes[0, 1].set_title(f'Box Plot of {column}')
    axes[0, 1].set_ylabel('Value')
    
    # Q-Q plot
    stats.probplot(df[column].dropna(), dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot')
    
    # Density plot
    df[column].plot(kind='density', ax=axes[1, 1])
    axes[1, 1].set_title(f'Density Plot of {column}')
    
    plt.tight_layout()
    return fig

# ============================================================================
# HELPER: CALCULATE COSTS
# ============================================================================

def calculate_api_cost(input_tokens, output_tokens):
    input_cost = (input_tokens / 1_000_000) * API_COSTS["per_1m_input"]
    output_cost = (output_tokens / 1_000_000) * API_COSTS["per_1m_output"]
    return input_cost + output_cost

def estimate_problem_cost():
    return calculate_api_cost(
        API_COSTS["avg_input_tokens"],
        API_COSTS["avg_output_tokens"]
    )

# ============================================================================
# AI PROBLEM SOLVER
# ============================================================================

def solve_problem_with_ai(problem_text, category, api_key, image_data=None, learning_mode=False):
    """Use Claude API to solve the problem"""
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
        
        system_prompt = f"""You are an expert statistics tutor in LEARNING MODE. 
Your goal is to help students UNDERSTAND the concepts, not just get answers.

IMPORTANT RULES:
1. ALWAYS show step-by-step work
2. EXPLAIN the WHY behind each step
3. Use LaTeX/math notation for formulas (wrap in $$)
4. When showing formulas use this format: $$formula$$
5. NEVER give just an answer - always include reasoning

Category: {category}
Learning Mode: {'Yes - Provide hints first, then solution' if learning_mode else 'Standard - Full solution with explanation'}

Format your response with these sections:
CONCEPT: What statistical concept applies here?
APPROACH: What method/formula should we use?
CALCULATION: Show the work step-by-step
ANSWER: The final result
INTERPRETATION: What does this mean in plain language?
ASSUMPTIONS: What assumptions did we make?"""
        
        text_prompt = f"""Solve this statistics problem step-by-step.

PROBLEM: {problem_text}

RULES:
- Show ALL calculations
- Use LaTeX for formulas: $$formula$$
- Explain each step
- State assumptions
- Give final answer with interpretation"""
        
        content.append({
            "type": "text",
            "text": text_prompt
        })

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
                "problem": problem_text[:50] if problem_text else "Image problem",
                "category": category,
                "cost": cost,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "has_image": image_data is not None,
                "learning_mode": learning_mode
            })
            
            return solution_text
        else:
            error_msg = f"API Error {response.status_code}"
            st.error(f"Error: {error_msg}")
            return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

# ============================================================================
# INITIALIZE SESSION STATE
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
# SIDEBAR: LOGIN & NAVIGATION
# ============================================================================

with st.sidebar:
    st.image("logo_1.png", width=250)
    st.divider()
    
    if not st.session_state.user_id:
        login_type = st.radio("Login as:", ["Student", "Admin"], key="login_type")
        
        if login_type == "Student":
            st.write("### Student Access")
            st.write("**Demo Credentials:**")
            st.write("Email: student@example.com")
            st.write("Password: password")
            st.write("")
            
            student_email = st.text_input("Email:", key="student_email", placeholder="student@example.com")
            student_pass = st.text_input("Password:", type="password", key="student_pass")
            
            # Terms acceptance
            terms_check = st.checkbox("I agree to the Terms of Service and Privacy Policy", key="terms_agree")
            
            if st.button("Sign In", key="student_signin"):
                if not terms_check:
                    st.error("Please accept the terms to continue")
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
            st.markdown("[Privacy Policy](https://jrgavillan.github.io/data_insight_studio/privacy_policy.md)")
        with col2:
            st.markdown("[Terms of Service](https://jrgavillan.github.io/data_insight_studio/privacy_policy.md)")
    
    else:
        st.write(f"### Welcome, {st.session_state.user_name}!")
        st.divider()
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Home", use_container_width=True):
                st.session_state.current_page = "home"
                st.rerun()
            if st.button("Analytics", use_container_width=True):
                st.session_state.current_page = "analytics"
                st.rerun()
        
        with col2:
            if st.button("Homework Help", use_container_width=True):
                st.session_state.current_page = "homework"
                st.rerun()
            if st.button("Resources", use_container_width=True):
                st.session_state.current_page = "resources"
                st.rerun()
        
        st.divider()
        
        if st.button("Sign Out", use_container_width=True):
            st.session_state.user_id = None
            st.session_state.user_name = None
            st.session_state.current_page = "home"
            st.rerun()

# ============================================================================
# MAIN CONTENT
# ============================================================================

if not st.session_state.user_id:
    st.error("Please sign in to continue")
else:
    current_page = st.session_state.current_page
    
    if current_page == "home":
        st.title("Data Insight Studio")
        st.subheader("AI-Powered Statistics Homework Helper")
        st.divider()
        
        st.write("""
        ### Welcome to Data Insight Studio!
        
        Get expert help with your statistics homework using AI!
        
        **Features:**
        - Homework Help - Upload images, Excel, CSV
        - Data Analysis - Descriptive stats, visualizations
        - Resources - Free study materials
        
        **Learning Features:**
        - Step-by-step explanations
        - Learning Mode - Hints first
        - Math formulas with LaTeX
        - Visualizations
        - Plain language interpretations
        
        **Pricing:** $14.99 per 90-day term
        
        **Academic Integrity:** Do NOT use on proctored exams
        
        Click Homework Help to get started!
        """)
        
        st.divider()
        
        st.subheader("Privacy & Legal")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("[Privacy Policy](https://jrgavillan.github.io/data_insight_studio/privacy_policy.md)")
        with col2:
            st.markdown("[Terms of Service](https://jrgavillan.github.io/data_insight_studio/privacy_policy.md)")
        with col3:
            st.markdown("[Contact Us](mailto:privacy@datainsightstudio.com)")
        
        st.info("We process and delete your uploads. We do not store them or use them for training.")
    
    elif current_page == "homework":
        st.header("HOMEWORK HELP")
        st.write("Upload an image, Excel file, CSV file, or type your problem below")
        st.divider()
        
        current_api_key = get_api_key()
        
        if not current_api_key:
            st.warning("System not configured yet. Please contact support.")
            st.info("Admin needs to configure API key before homework help is available.")
        else:
            learning_mode = st.checkbox("Learning Mode (hints first, then solution)", value=False)
            if learning_mode:
                st.info("In Learning Mode, you'll get hints first to help you work through the problem yourself!")
            
            col1, col2 = st.columns(2)
            with col1:
                category = st.selectbox("Category:", 
                    ["Descriptive Statistics", "Hypothesis Testing", "Confidence Intervals", 
                     "Probability", "Regression", "ANOVA", "T-Tests", "Other"], key="cat")
            with col2:
                difficulty = st.selectbox("Difficulty:", ["Beginner", "Intermediate", "Advanced"], key="diff")
            
            st.write("")
            
            st.write("### Upload File (Optional)")
            st.write("Image, Excel, or CSV - whatever your homework is!")
            
            uploaded_file = st.file_uploader(
                "Choose a file:",
                type=["jpg", "jpeg", "png", "xlsx", "xls", "csv"],
                key="problem_file"
            )
            
            data_analysis_mode = False
            df = None
            
            if uploaded_file:
                if uploaded_file.name.endswith(('jpg', 'jpeg', 'png')):
                    st.image(uploaded_file, caption="Your homework problem", use_container_width=True)
                elif uploaded_file.name.endswith(('.csv', '.xlsx', '.xls')):
                    st.info(f"File uploaded: {uploaded_file.name}")
                    df = read_excel_csv(uploaded_file)
                    
                    if df is not None:
                        st.write("**File Preview:**")
                        st.dataframe(df.head(), use_container_width=True)
                        
                        data_analysis_mode = st.checkbox("Analyze this data")
                        
                        if data_analysis_mode:
                            st.write("### Quick Analysis")
                            
                            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                            
                            if numeric_cols:
                                col_to_analyze = st.selectbox("Select column to analyze:", numeric_cols)
                                
                                st.write("**Descriptive Statistics:**")
                                stats_dict = calculate_descriptive_stats(df, col_to_analyze)
                                stats_df = pd.DataFrame(list(stats_dict.items()), columns=['Statistic', 'Value'])
                                st.dataframe(stats_df, use_container_width=True)
                                
                                st.write("**Visualizations:**")
                                fig = create_visualizations(df, col_to_analyze)
                                st.pyplot(fig)
            
            st.write("")
            st.write("### Or Type Your Problem")
            st.write("(Optional - you can also just upload a file)")
            
            problem = st.text_area(
                "Your problem:",
                placeholder="Type your problem OR upload a file above...",
                height=100
            )
            
            st.write("")
            
            st.warning("REMINDER: Do NOT use this on proctored exams. This is for learning and homework preparation only.")
            
            if st.button("SOLVE THIS & LEARN", use_container_width=True):
                problem_text_final = problem.strip() if problem else ""
                image_b64 = None
                
                if uploaded_file:
                    if uploaded_file.name.endswith(('jpg', 'jpeg', 'png')):
                        image_b64 = image_to_base64(uploaded_file)
                    elif uploaded_file.name.endswith(('.csv', '.xlsx', '.xls')):
                        if not data_analysis_mode:
                            problem_text_final = f"Analyze this data file:\n\n{df.to_string()}\n\n{problem_text_final}"
                
                if problem_text_final or image_b64:
                    with st.spinner("AI is solving and explaining..."):
                        solution = solve_problem_with_ai(
                            problem_text_final,
                            category,
                            current_api_key,
                            image_data=image_b64,
                            learning_mode=learning_mode
                        )
                    
                    if solution:
                        st.divider()
                        st.subheader("STEP-BY-STEP SOLUTION")
                        st.markdown(solution)
                        
                        st.markdown("*This solution uses LaTeX notation for mathematical formulas.*")
                else:
                    st.error("Enter your problem or upload a file")
    
    elif current_page == "analytics":
        st.header("Analytics")
        st.info("Coming soon - Track your learning progress!")
    
    elif current_page == "resources":
        st.header("Resources")
        st.subheader("Free Study Materials")
        st.divider()
        
        tab1, tab2, tab3, tab4 = st.tabs(["Formulas", "Guides", "Tutorials", "FAQ"])
        
        with tab1:
            st.write("**Common Statistical Formulas:**")
            st.markdown("""
            - Mean: $$\\mu = \\frac{\\sum x}{n}$$
            - Variance: $$\\sigma^2 = \\frac{\\sum(x - \\mu)^2}{n}$$
            - Standard Deviation: $$\\sigma = \\sqrt{\\sigma^2}$$
            - Z-score: $$z = \\frac{x - \\mu}{\\sigma}$$
            - t-statistic: $$t = \\frac{\\bar{x} - \\mu_0}{s/\\sqrt{n}}$$
            """)
        
        with tab2:
            st.write("**Study Guides:**")
            st.write("- Introduction to Statistics")
            st.write("- Probability Basics")
            st.write("- Hypothesis Testing 101")
        
        with tab3:
            st.write("**Video Tutorials:**")
            st.write("- Understanding Normal Distribution")
            st.write("- T-Tests Explained")
            st.write("- Regression Analysis Guide")
        
        with tab4:
            st.write("**FAQ:**")
            st.write("**Q: Will this help me cheat?**")
            st.write("A: No - we focus on LEARNING, not answers.")
            st.write("**Q: Can I use this on exams?**")
            st.write("A: NO - Do not use on proctored exams.")
            st.write("**Q: Is my data safe?**")
            st.write("A: Yes - we delete uploads immediately.")
    
    # ADMIN DASHBOARD
    if st.session_state.user_id == "admin":
        st.divider()
        st.header("ADMIN DASHBOARD")
        st.divider()
        
        st.subheader("API Configuration")
        api_key = st.text_input(
            "Paste your Claude API key:",
            type="password",
            key="admin_api_key",
            placeholder="sk-ant-...",
            value=load_api_key() or ""
        )
        
        if api_key:
            if save_api_key(api_key):
                st.session_state.api_key = api_key
                st.success("API key saved!")
            else:
                st.error("Could not save API key")
        
        st.divider()
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total API Calls", st.session_state.api_calls)
        col2.metric("Total API Cost", f"${st.session_state.api_costs:.2f}")
        col3.metric("Avg Cost/Problem", f"${estimate_problem_cost():.3f}")
        if PRICING['per_term'] > 0:
            profit = ((PRICING['per_term'] - st.session_state.api_costs) / PRICING['per_term'] * 100)
            col4.metric("Profit Margin", f"{profit:.1f}%")
        
        st.divider()
        
        st.subheader("USAGE LOG")
        if st.session_state.usage_log:
            log_df = pd.DataFrame(st.session_state.usage_log)
            st.dataframe(log_df[["timestamp", "category", "learning_mode", "has_image", "cost"]], use_container_width=True)
        else:
            st.info("No API calls yet")
