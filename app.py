import streamlit as st
import pickle
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

# Custom CSS for gorgeous UI, soft shadows, and card styling
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #4F46E5;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.6rem;
        border: none;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #4338CA;
    }
    .prediction-card {
        padding: 24px;
        border-radius: 12px;
        background: white;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        text-align: center;
        margin-top: 20px;
        border: 1px solid #E5E7EB;
    }
    </style>
""", unsafe_allow_html=True)

# Load the trained KNN model
@st.cache_resource
def load_model():
    with open("knn.pkl", "rb") as f:
        model = pickle.load(f)
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model file (`knn.pkl`): {e}")
    st.stop()

# Header Section
st.title("🎓 Student Performance Predictor")
st.markdown("Enter the student details below to generate real-time performance predictions using your trained **K-Nearest Neighbors Regressor** model.")
st.markdown("---")

# Form Layout
with st.form("prediction_form"):
    st.markdown("### 📋 Student Parameter Inputs")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age[cite: 1]", min_value=15, max_value=50, value=20, step=1)
        gender = st.selectbox("Gender[cite: 1]", ["Male", "Female", "Other"])
        course = st.selectbox("Course[cite: 1]", ["Computer Science", "Engineering", "Business", "Arts", "Science"])
        study_hours = st.slider("Study Hours (per day)[cite: 1]", 0.0, 16.0, 5.0, 0.5)
        class_attendance = st.slider("Class Attendance (%)[cite: 1]", 0, 100, 85)
        internet_access = st.selectbox("Internet Access[cite: 1]", ["Yes", "No"])
        
    with col2:
        sleep_hours = st.slider("Sleep Hours (per night)[cite: 1]", 2.0, 12.0, 7.0, 0.5)
        sleep_quality = st.selectbox("Sleep Quality[cite: 1]", ["Poor", "Average", "Good", "Excellent"])
        study_method = st.selectbox("Study Method[cite: 1]", ["Self-Study", "Online Videos", "Group Study", "Classes"])
        facility_rating = st.slider("Facility Rating (1-5)[cite: 1]", 1, 5, 4)
        exam_difficulty = st.selectbox("Exam Difficulty[cite: 1]", ["Easy", "Medium", "Hard"])
    
    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
    submit_button = st.form_submit_button(label="Calculate Prediction")

# Prediction Execution and Output Card
if submit_button:
    # Encoding categorical fields to match standard model requirements
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
            <div class="prediction-card">
                <h4 style="color: #4F46E5; margin-bottom: 5px; text-transform: uppercase; font-size: 0.85rem; letter-spacing: 0.05em;">Prediction Result</h4>
                <h1 style="color: #1F2937; font-size: 2.8rem; margin: 10px 0;">{prediction:.2f}</h1>
                <p style="color: #6B7280; font-size: 0.95rem;">Estimated outcome score generated successfully.</p>
            </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
