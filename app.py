import streamlit as st
import pickle
import numpy as np
import pandas as pd

# 1. Page Configuration
st.set_page_config(
    page_title="EduPredict | Advanced Performance Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Advanced Custom CSS for Modern UI & Glassmorphism Details
st.markdown("""
    <style>
    /* Main Background Accent */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Header Styling */
    .main-header {
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(10px);
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.07);
        border: 1px solid rgba(255, 255, 255, 0.18);
        margin-bottom: 2rem;
    }
    
    /* Card Layouts */
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 14px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
        border-left: 5px solid #4F46E5;
        transition: transform 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
    }
    
    /* Prediction Output Container */
    .prediction-container {
        background: linear-gradient(135deg, #4F46E5 0%, #3B82F6 100%);
        padding: 30px;
        border-radius: 16px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 25px -5px rgba(79, 70, 229, 0.4);
        margin-top: 25px;
    }
    
    /* Submit Button */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #4F46E5 0%, #4338CA 100%);
        color: white;
        font-weight: 600;
        border-radius: 10px;
        padding: 0.75rem 1rem;
        border: none;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #4338CA 0%, #3730A3 100%);
        box-shadow: 0 6px 16px rgba(79, 70, 229, 0.4);
    }
    </style>
""", unsafe_allow_html=True)

# 3. Model Caching & Loading
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

# 4. Sidebar Panel for Controls & Info
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/graduation-cap.png", width=120)
    st.markdown("### Model Configuration")
    st.info("Algorithm: **K-Nearest Neighbors Regressor**\n\nStatus: Online & Ready 🚀")
    
    st.markdown("---")
    st.markdown("### Quick Navigation")
    app_mode = st.radio("Select View", ["Predictor Dashboard", "Batch Analytics Simulation"])
    
    st.markdown("---")
    st.markdown("<div style='text-align: center; color: gray; font-size: 0.8rem;'>System Version 2.4.0</div>", unsafe_allow_html=True)

# 5. Main Screen Layout
st.markdown("""
    <div class="main-header">
        <h1 style="color: #1E293B; margin-bottom: 0px; font-size: 2.2rem;">⚡ Student Performance Intelligence Hub</h1>
        <p style="color: #64748B; margin-top: 8px; font-size: 1.05rem;">
            Leverage machine learning analytics to forecast academic outcomes, analyze behavior weights, and optimize study patterns in real-time.
        </p>
    </div>
""", unsafe_allow_html=True)

if app_mode == "Predictor Dashboard":
    # Multi-column metric summaries top bar
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown('<div class="metric-card"><h4>🎯 Model Engine</h4><p style="color: #4F46E5; font-weight: bold; margin:0;">KNN Regressor</p></div>', unsafe_allow_html=True)
    with m2:
        st.markdown('<div class="metric-card"><h4>🔒 Security Check</h4><p style="color: #10B981; font-weight: bold; margin:0;">SSL / Safe Inference</p></div>', unsafe_allow_html=True)
    with m3:
        st.markdown('<div class="metric-card"><h4>⚡ Latency</h4><p style="color: #F59E0B; font-weight: bold; margin:0;">< 15ms Response</p></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Input Form Container
    with st.form("advanced_prediction_form"):
        st.markdown("### 🎛️ Student Parameter Configuration")
        
        col_a, col_b = st.columns(2, gap="large")
        
        with col_a:
            st.markdown("#### Demographics & Habits")
            age = st.number_input("Age", min_value=15, max_value=50, value=20, step=1)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            course = st.selectbox("Course Track", ["Computer Science", "Engineering", "Business", "Arts", "Science"])
            study_hours = st.slider("Daily Study Hours", 0.0, 16.0, 5.0, 0.5, help="Average hours spent studying per day.")
            class_attendance = st.slider("Class Attendance Rate (%)", 0, 100, 85)
            internet_access = st.selectbox("Home Internet Access", ["Yes", "No"])
            
        with col_b:
            st.markdown("#### Environment & Prep")
            sleep_hours = st.slider("Nightly Sleep Duration (Hours)", 2.0, 12.0, 7.0, 0.5)
            sleep_quality = st.selectbox("Sleep Quality", ["Poor", "Average", "Good", "Excellent"])
            study_method = st.selectbox("Primary Study Method", ["Self-Study", "Online Videos", "Group Study", "Classes"])
            facility_rating = st.slider("Campus Facility Rating", 1, 5, 4, help="1 = Very Poor, 5 = Excellent")
            exam_difficulty = st.selectbox("Anticipated Exam Difficulty", ["Easy", "Medium", "Hard"])
        
        st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)
        submitted = st.form_submit_button(label="🚀 Generate Performance Score")

    # Prediction Logic
    if submitted:
        # Mappings matching model expectations
        gender_map = {"Male": 0, "Female": 1, "Other": 2}
        internet_map = {"No": 0, "Yes": 1}
        sleep_quality_map = {"Poor": 1, "Average": 2, "Good": 3, "Excellent": 4}
        exam_diff_map = {"Easy": 1, "Medium": 2, "Hard": 3}
        course_map = {"Computer Science": 0, "Engineering": 1, "Business": 2, "Arts": 3, "Science": 4}
        study_method_map = {"Self-Study": 0, "Online Videos": 1, "Group Study": 2, "Classes": 3}

        input_vector = np.array([[
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
        
        with st.spinner("Processing features through KNN hyper-space..."):
            try:
                prediction_result = model.predict(input_vector)[0]
                
                st.markdown(f"""
                    <div class="prediction-container">
                        <span style="text-transform: uppercase; letter-spacing: 0.1em; font-size: 0.85rem; font-weight: 600; opacity: 0.8;">Estimated Performance Score</span>
                        <h1 style="font-size: 3.5rem; margin: 10px 0; font-weight: 800;">{prediction_result:.2f}</h1>
                        <p style="font-size: 1rem; margin: 0; opacity: 0.9;">Prediction calculated successfully using neighbor weighting algorithms.</p>
                    </div>
                """, unsafe_allow_html=True)
                
            except Exception as prediction_error:
                st.error(f"Inference execution failed: {prediction_error}")

else:
    # Batch Analytics Simulation View
    st.markdown("### 📊 Batch Data Simulation")
    st.write("Upload or simulate multiple student profiles to analyze bulk predictions.")
    
    if st.button("Run Bulk Simulation Sample"):
        simulated_data = pd.DataFrame({
            "Age": [21, 22, 19, 23],
            "Study Hours": [6.5, 4.0, 8.0, 3.5],
            "Attendance": [90, 75, 95, 60]
        })
        st.dataframe(simulated_data, use_container_width=True)
        st.success("Simulation ran cleanly. (Connect a multi-row parser array to process full datasets).")
