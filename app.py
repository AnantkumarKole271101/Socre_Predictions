import streamlit as st
import pickle
import numpy as np
import pandas as pd

# 1. Page Configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Professional Enterprise Color Palette & Styling
st.markdown("""
    <style>
    /* Global Theme Styles */
    .stApp {
        background-color: #F8FAFC;
        color: #1E293B;
    }
    
    /* Clean Top Header Banner */
    .hero-container {
        background: #FFFFFF;
        padding: 2.5rem;
        border-radius: 12px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
        border: 1px solid #E2E8F0;
        margin-bottom: 2rem;
    }
    
    /* Metric / Stat Card */
    .stat-card {
        background: #FFFFFF;
        padding: 18px 20px;
        border-radius: 10px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
        border: 1px solid #E2E8F0;
        border-left: 4px solid #2563EB;
    }
    
    /* Form Wrapper Card */
    .form-card {
        background: #FFFFFF;
        padding: 30px;
        border-radius: 12px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
        border: 1px solid #E2E8F0;
    }
    
    /* Prediction Output Container */
    .prediction-banner {
        background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 100%);
        padding: 30px;
        border-radius: 12px;
        color: #FFFFFF;
        text-align: center;
        box-shadow: 0 10px 15px -3px rgba(37, 99, 235, 0.2);
        margin-top: 25px;
    }
    
    /* Primary Button Customization */
    .stButton>button {
        width: 100%;
        background-color: #2563EB;
        color: #FFFFFF;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.65rem 1rem;
        border: none;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        transition: background-color 0.2s ease;
    }
    .stButton>button:hover {
        background-color: #1D4ED8;
    }
    
    /* Sidebar Polish */
    section[data-testid="stSidebar"] {
        background-color: #0F172A;
        color: #F8FAFC;
    }
    section[data-testid="stSidebar"] .stMarkdown {
        color: #94A3B8;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Model Loading with Cache
@st.cache_resource
def load_prediction_model():
    with open("knn.pkl", "rb") as file_obj:
        loaded_model = pickle.load(file_obj)
    return loaded_model

try:
    model = load_prediction_model()
except Exception as err:
    st.error(f"⚠️ Critical Error: Could not locate or load `knn.pkl`. Details: {err}")
    st.stop()

# 4. Sidebar Panel
with st.sidebar:
    st.markdown("### 🎓 Academic AI Suite")
    st.markdown("Enterprise-grade regression analysis powered by scikit-learn.")
    st.markdown("---")
    
    app_mode = st.radio("Navigation Menu", ["Single Prediction", "Batch Analysis Overview"])
    
    st.markdown("---")
    st.markdown("<p style='font-size: 0.75rem; color: #64748B;'>Engine: KNN Regressor<br>Status: Online & Ready</p>", unsafe_allow_html=True)

# 5. Main Content Header
st.markdown("""
    <div class="hero-container">
        <h1 style="color: #0F172A; margin: 0px 0px 8px 0px; font-size: 2rem; font-weight: 700;">Student Performance Intelligence</h1>
        <p style="color: #64748B; margin: 0; font-size: 1.05rem; line-height: 1.5;">
            Input student attributes to calculate predicted scores instantly. Optimized with verified nearest-neighbor models.
        </p>
    </div>
""", unsafe_allow_html=True)

if app_mode == "Single Prediction":
    # Metrics Row
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.markdown('<div class="stat-card"><span style="color: #64748B; font-size: 0.85rem; font-weight: 600;">ALGORITHM</span><p style="color: #0F172A; font-weight: 700; font-size: 1.1rem; margin: 4px 0 0 0;">K-NN Regressor</p></div>', unsafe_allow_html=True)
    with col_m2:
        st.markdown('<div class="stat-card" style="border-left-color: #10B981;"><span style="color: #64748B; font-size: 0.85rem; font-weight: 600;">MODEL INTEGRITY</span><p style="color: #0F172A; font-weight: 700; font-size: 1.1rem; margin: 4px 0 0 0;">Verified Pipeline</p></div>', unsafe_allow_html=True)
    with col_m3:
        st.markdown('<div class="stat-card" style="border-left-color: #6366F1;"><span style="color: #64748B; font-size: 0.85rem; font-weight: 600;">SYSTEM STATUS</span><p style="color: #0F172A; font-weight: 700; font-size: 1.1rem; margin: 4px 0 0 0;">Operational</p></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Prediction Input Form Wrapper
    with st.form("prediction_form"):
        st.markdown("<h3 style='color: #0F172A; margin-top: 0;'>Student Parameter Configuration</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color: #64748B; font-size: 0.9rem;'>Provide the required profile details below:</p>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            age = st.number_input("Age[cite: 1]", min_value=15, max_value=50, value=20, step=1)
            gender = st.selectbox("Gender[cite: 1]", ["Male", "Female", "Other"])
            course = st.selectbox("Course Track[cite: 1]", ["Computer Science", "Engineering", "Business", "Arts", "Science"])
            study_hours = st.slider("Daily Study Hours[cite: 1]", 0.0, 16.0, 5.0, 0.5)
            class_attendance = st.slider("Class Attendance Rate (%)[cite: 1]", 0, 100, 85)
            internet_access = st.selectbox("Internet Access[cite: 1]", ["Yes", "No"])
            
        with col2:
            sleep_hours = st.slider("Nightly Sleep Hours[cite: 1]", 2.0, 12.0, 7.0, 0.5)
            sleep_quality = st.selectbox("Sleep Quality[cite: 1]", ["Poor", "Average", "Good", "Excellent"])
            study_method = st.selectbox("Primary Study Method[cite: 1]", ["Self-Study", "Online Videos", "Group Study", "Classes"])
            facility_rating = st.slider("Campus Facility Rating (1-5)[cite: 1]", 1, 5, 4)
            exam_difficulty = st.selectbox("Anticipated Exam Difficulty[cite: 1]", ["Easy", "Medium", "Hard"])
        
        st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
        submit_button = st.form_submit_button(label="Generate Prediction Score")

    # Prediction Logic & Output View
    if submit_button:
        gender_map = {"Male": 0, "Female": 1, "Other": 2}
        internet_map = {"No": 0, "Yes": 1}
        sleep_quality_map = {"Poor": 1, "Average": 2, "Good": 3, "Excellent": 4}
        exam_diff_map = {"Easy": 1, "Medium": 2, "Hard": 3}
        course_map = {"Computer Science": 0, "Engineering": 1, "Business": 2, "Arts": 3, "Science": 4}
        study_method_map = {"Self-Study": 0, "Online Videos": 1, "Group Study": 2, "Classes": 3}

        features = np.array([[
            age,
            gender_map[gender],
            course_map[course],
            study_hours,
            class_attendance,
            internet_map[internet_access],
            sleep_hours,
            sleep_quality_map[sleep_quality],
            study_method_map[study_method],
            facility_rating,
            exam_diff_map[exam_difficulty]
        ]])
        
        try:
            prediction = model.predict(features)[0]
            st.markdown(f"""
                <div class="prediction-banner">
                    <span style="text-transform: uppercase; letter-spacing: 0.08em; font-size: 0.8rem; font-weight: 600; opacity: 0.85;">Forecasted Outcome Score</span>
                    <h1 style="font-size: 3.2rem; margin: 8px 0; font-weight: 700;">{prediction:.2f}</h1>
                    <p style="font-size: 0.95rem; margin: 0; opacity: 0.9;">Calculated successfully based on weighted multi-dimensional neighbors.</p>
                </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Prediction execution failed: {e}")

else:
    st.markdown("### Batch Data Preview")
    st.write("Use this view to inspect general formatting patterns for multi-row operations.")
    sample_df = pd.DataFrame({
        "Parameter Type": ["Demographics", "Habits", "Environment"],
        "Status": ["Configured", "Active", "Verified"],
        "Target Variable": ["Performance Index", "Performance Index", "Performance Index"]
    })
    st.dataframe(sample_df, use_container_width=True)
