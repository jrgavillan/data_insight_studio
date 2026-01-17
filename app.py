import streamlit as st
import requests
import base64
import json
import os
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

st.set_page_config(page_title="Data Insight Studio", layout="wide")

CONFIG_FILE = "api_config.txt"

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
    # Try environment variable first (Render/cloud)
    env_key = os.environ.get("ANTHROPIC_API_KEY")
    if env_key:
        return env_key
    
    # Fall back to file (local development)
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
    
    # Try to load from file
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

def solve_problem_with_ai(problem_text, category, api_key, image_data=None):
    """Use Claude API to solve the problem"""
    try:
        # Validate API key
        if not api_key:
            st.error("❌ API key is missing!")
            return None
        
        if not api_key.startswith("sk-ant-"):
            st.error("❌ Invalid API key format. Key should start with 'sk-ant-'")
            return None
        
        # Validate problem input
        if not image_data and (not problem_text or problem_text.strip() == ""):
            st.error("❌ Please enter a problem or upload an image")
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
            content.append({
                "type": "text",
                "text": f"""Please read this homework problem from the image and solve it step-by-step.

Category: {category}

If there's also text below, use both the image and text to understand the problem.

Problem text: {problem_text if problem_text else "See image above"}

Provide a complete solution with:
1. ANALYSIS: What the problem asks
2. CONCEPT: Statistical concepts
3. STEPS: Step-by-step solution
4. ANSWER: Final answer
5. INTERPRETATION: What it means

Format exactly:
ANALYSIS: [text]
CONCEPT: [text]
STEPS: 
- Step 1: [text]
- Step 2: [text]
ANSWER: [text]
INTERPRETATION: [text]"""
            })
        else:
            content.append({
                "type": "text",
                "text": f"""You are a statistics tutor solving this problem:

PROBLEM: {problem_text}
CATEGORY: {category}

Provide a complete solution with:
1. ANALYSIS: What the problem asks
2. CONCEPT: Statistical concepts
3. STEPS: Step-by-step solution
4. ANSWER: Final answer
5. INTERPRETATION: What it means

Format exactly:
ANALYSIS: [text]
CONCEPT: [text]
STEPS: 
- Step 1: [text]
- Step 2: [text]
ANSWER: [text]
INTERPRETATION: [text]"""
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
                "max_tokens": 1500,
                "messages": [{"role": "user", "content": content}]
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            solution_text = result['content'][0]['text']
            
            # Track usage
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
                "has_image": image_data is not None
            })
            
            return parse_solution(solution_text)
        else:
            # Better error messages for debugging
            error_msg = f"API Error {response.status_code}"
            try:
                error_detail = response.json()
                if "error" in error_detail:
                    error_msg += f": {error_detail['error'].get('message', 'Unknown error')}"
            except:
                error_msg += f": {response.text[:200]}"
            
            st.error(f"❌ {error_msg}")
            
            if response.status_code == 400:
                st.warning("⚠️ **Troubleshooting 400 Error:**")
                st.write("- Verify API key is correct (starts with sk-ant-)")
                st.write("- Check that you have API credits")
                st.write("- Try a simpler problem text")
                st.write("- Refresh page and try again")
            elif response.status_code == 401:
                st.error("❌ Invalid API key. Double-check your credentials.")
            elif response.status_code == 429:
                st.error("❌ Rate limit exceeded. Please wait a moment and try again.")
            
            return None
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.info("💡 Try refreshing the page or checking your internet connection.")
        return None

def parse_solution(text):
    """Parse solution text into structured format"""
    solution = {
        "analysis": "",
        "concept": "",
        "steps": [],
        "answer": "",
        "interpretation": ""
    }
    
    sections = {
        "ANALYSIS:": "analysis",
        "CONCEPT:": "concept",
        "STEPS:": "steps",
        "ANSWER:": "answer",
        "INTERPRETATION:": "interpretation"
    }
    
    current_section = None
    current_content = []
    
    for line in text.split('\n'):
        matched = False
        for marker, key in sections.items():
            if marker in line:
                if current_section and current_content:
                    if current_section == "steps":
                        solution[current_section] = [s.strip('- ').strip() for s in current_content if s.strip()]
                    else:
                        solution[current_section] = ' '.join(current_content).strip()
                current_section = key
                current_content = [line.replace(marker, '').strip()]
                matched = True
                break
        
        if not matched and current_section:
            current_content.append(line.strip())
    
    if current_section and current_content:
        if current_section == "steps":
            solution[current_section] = [s.strip('- ').strip() for s in current_content if s.strip()]
        else:
            solution[current_section] = ' '.join(current_content).strip()
    
    return solution

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

# ============================================================================
# SIDEBAR: LOGIN & NAVIGATION
# ============================================================================

with st.sidebar:
    st.title("📊 Data Insight Studio")
    st.divider()
    
    if not st.session_state.user_id:
        # LOGIN FORM
        login_type = st.radio("Login as:", ["Student", "Admin"], key="login_type")
        
        if login_type == "Student":
            st.write("### Student Access")
            st.write("**Demo Credentials:**")
            st.write("Email: student@example.com")
            st.write("Password: password")
            st.write("")
            
            student_email = st.text_input("Email:", key="student_email", placeholder="student@example.com")
            student_pass = st.text_input("Password:", type="password", key="student_pass", placeholder="password")
            
            if st.button("Sign In", key="student_signin"):
                if student_email == "student@example.com" and student_pass == "password":
                    st.session_state.user_id = f"student_{student_email}"
                    st.session_state.user_name = "Student"
                    st.rerun()
                else:
                    st.error("❌ Invalid email or password. Use student@example.com / password")
        
        else:  # Admin
            st.write("### Admin Access")
            admin_pass = st.text_input("Password:", type="password", key="admin_pass")
            
            if st.button("Sign In", key="admin_signin"):
                if admin_pass == "admin123":
                    st.session_state.user_id = "admin"
                    st.session_state.user_name = "Admin"
                    st.rerun()
                else:
                    st.error("❌ Invalid password")
    
    else:
        # LOGGED IN - NAVIGATION
        st.write(f"### Welcome, {st.session_state.user_name}! 👋")
        st.divider()
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🏠 Home", use_container_width=True):
                st.session_state.current_page = "home"
                st.rerun()
            if st.button("📊 Analytics", use_container_width=True):
                st.session_state.current_page = "analytics"
                st.rerun()
        
        with col2:
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
        # HOME PAGE
        st.title("📊 Data Insight Studio")
        st.subheader("AI-Powered Homework Helper")
        st.divider()
        
        st.write("""
        ### Welcome to Data Insight Studio! 🎓
        
        Get instant help with your statistics homework using AI!
        
        **Features:**
        - 📚 **Homework Help** - Upload homework images → Get AI solutions
        - 📊 **Analytics** - Coming soon
        - 📈 **Resources** - Free study materials
        
        **Pricing:** $14.99 per 90-day term
        
        Click **📚 Homework Help** to get started!
        """)
    
    elif current_page == "homework":
        # HOMEWORK HELP (PAID FEATURE)
        st.header("📚 HOMEWORK HELP")
        st.write("Upload an image or type your problem below")
        st.divider()
        
        # Get the API key
        current_api_key = get_api_key()
        
        if not current_api_key:
            st.warning("⚠️ System not configured yet. Please contact support.")
            st.info("Admin needs to configure API key before homework help is available.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                category = st.selectbox("Category:", 
                    ["Descriptive Statistics", "Hypothesis Testing", "Confidence Intervals", 
                     "Probability", "Regression", "Other"], key="cat")
            with col2:
                difficulty = st.selectbox("Difficulty:", ["Beginner", "Intermediate", "Advanced"], key="diff")
            
            st.write("")
            
            # FILE UPLOAD (Images only)
            st.write("### 📸 Upload Image (Optional)")
            st.write("Take a photo or upload an image of your homework")
            
            uploaded_image = st.file_uploader(
                "Choose an image:",
                type=["jpg", "jpeg", "png"],
                key="problem_image"
            )
            
            if uploaded_image:
                st.image(uploaded_image, caption="Your homework problem", use_container_width=True)
            
            st.write("")
            st.write("### 📝 Or Type Your Problem")
            st.write("(Optional - you can also just upload an image)")
            
            problem = st.text_area(
                "Your problem:",
                placeholder="Type your problem OR upload an image above...",
                height=100
            )
            
            st.write("")
            
            if st.button("🔍 SOLVE THIS", use_container_width=True):
                problem_text_final = problem.strip() if problem else ""
                image_b64 = None
                
                if uploaded_image:
                    image_b64 = image_to_base64(uploaded_image)
                
                if problem_text_final or image_b64:
                    with st.spinner("🤔 AI is solving..."):
                        solution = solve_problem_with_ai(
                            problem_text_final,
                            category,
                            current_api_key,
                            image_data=image_b64
                        )
                    
                    if solution:
                        st.divider()
                        st.subheader("✅ SOLUTION")
                        
                        st.write(f"**Analysis:** {solution['analysis']}")
                        st.write(f"**Concept:** {solution['concept']}")
                        st.write("**Steps:**")
                        for i, s in enumerate(solution['steps'], 1):
                            st.write(f"{i}. {s}")
                        st.success(f"**Answer:** {solution['answer']}")
                        st.info(f"**Interpretation:** {solution['interpretation']}")
                else:
                    st.error("Enter your problem or upload an image")
    
    elif current_page == "analytics":
        st.header("📊 Analytics")
        st.info("🔄 Coming soon - Track your learning progress with detailed analytics!")
    
    elif current_page == "resources":
        st.header("📈 Resources")
        st.subheader("Free Study Materials")
        st.divider()
        
        tab1, tab2, tab3 = st.tabs(["Formulas", "Guides", "Tutorials"])
        
        with tab1:
            st.write("**Common Statistical Formulas:**")
            st.write("- Mean: μ = Σx / n")
            st.write("- Variance: σ² = Σ(x - μ)² / n")
            st.write("- Standard Deviation: σ = √variance")
            st.write("- Z-score: z = (x - μ) / σ")
            st.write("More formulas coming soon...")
        
        with tab2:
            st.write("**Study Guides:**")
            st.write("- Introduction to Statistics")
            st.write("- Probability Basics")
            st.write("- Hypothesis Testing 101")
            st.write("More guides coming soon...")
        
        with tab3:
            st.write("**Video Tutorials:**")
            st.write("- Understanding Normal Distribution")
            st.write("- T-Tests Explained")
            st.write("- Regression Analysis Guide")
            st.write("More tutorials coming soon...")
    
    # ADMIN DASHBOARD
    if st.session_state.user_id == "admin":
        st.divider()
        st.header("📊 ADMIN DASHBOARD")
        st.divider()
        
        # Show API key input
        st.subheader("🔑 API Configuration")
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
                st.success("✅ API key saved! (Will persist even after logout)")
            else:
                st.error("❌ Could not save API key")
        
        st.divider()
        
        # Analytics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total API Calls", st.session_state.api_calls)
        col2.metric("Total API Cost", f"${st.session_state.api_costs:.2f}")
        col3.metric("Avg Cost/Problem", f"${estimate_problem_cost():.3f}")
        if PRICING['per_term'] > 0:
            profit = ((PRICING['per_term'] - st.session_state.api_costs) / PRICING['per_term'] * 100)
            col4.metric("Profit Margin", f"{profit:.1f}%")
        
        st.divider()
        
        st.subheader("📈 USAGE LOG")
        if st.session_state.usage_log:
            # Simple display without pandas
            st.write("**Recent API Calls:**")
            for log in st.session_state.usage_log[-10:]:
                st.write(f"- {log['timestamp'].strftime('%H:%M:%S')} | {log['category']} | ${log['cost']:.4f}")
        else:
            st.info("No API calls yet")
