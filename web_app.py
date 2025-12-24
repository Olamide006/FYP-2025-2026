# ============================================================
# DIABETES PREDICTION - WEB APPLICATION
# Interactive interface for diabetes risk prediction
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings

warnings.filterwarnings('ignore')

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Diabetes Risk Prediction",
    page_icon="🏥",
    layout="wide"
)

# ============================================================
# LOAD MODELS AND DATA
# ============================================================
@st.cache_resource
def load_models():
    """Load saved models and encoders"""
    try:
        model = joblib.load('best_model.pkl')
        scaler = joblib.load('scaler.pkl')
        label_encoders = joblib.load('label_encoders.pkl')
        return model, scaler, label_encoders
    except FileNotFoundError:
        st.error("Model files not found! Please run 2_train_models.py first.")
        st.stop()

model, scaler, label_encoders = load_models()

# Load feature names
try:
    feature_names = pd.read_csv('feature_names.csv')['features'].tolist()
except FileNotFoundError:
    st.error("feature_names.csv not found! Please run 2_train_models.py first.")
    st.stop()

# ============================================================
# HEADER
# ============================================================
st.title("🏥 Diabetes Risk Prediction System")
st.markdown("""
This application predicts diabetes risk using machine learning based on patient health information.
**Note:** This is a screening tool and not a substitute for professional medical diagnosis.
""")

st.markdown("---")

# ============================================================
# SIDEBAR - INFORMATION
# ============================================================
st.sidebar.header("ℹ️ About")
st.sidebar.info("""
**Diabetes Risk Prediction System**

This tool uses machine learning to assess diabetes risk based on:
- Demographics
- Physical measurements
- Lab test results
- Medical history

Developed as part of a Final Year Project at the University of Ilorin.
""")

st.sidebar.markdown("---")
st.sidebar.header("📊 Model Performance")

# Load and display model results
try:
    results = pd.read_csv('model_results.csv')
    best_model_name = results.iloc[0]['Model']
    best_accuracy = results.iloc[0]['Accuracy']
    
    st.sidebar.success(f"**Best Model:** {best_model_name}")
    st.sidebar.metric("Accuracy", f"{best_accuracy*100:.2f}%")
    
    with st.sidebar.expander("View All Models"):
        st.dataframe(results[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']])
except FileNotFoundError:
    pass

# ============================================================
# MAIN CONTENT - INPUT FORM
# ============================================================
st.header("📝 Patient Information")

# Create tabs for better organization
tab1, tab2, tab3 = st.tabs(["Demographics", "Lab Results", "Medical History"])

# Initialize input dictionary
input_data = {}

# TAB 1: Demographics
with tab1:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=30, step=1)
        input_data['age'] = age
    
    with col2:
        sex = st.selectbox("Sex", options=["Male", "Female"])
        input_data['sex'] = sex
    
    with col3:
        residence = st.selectbox("Residence", options=["Urban", "Rural"])
        input_data['residence'] = residence
    
    col4, col5, col6 = st.columns(3)
    
    with col4:
        height = st.number_input("Height (cm)", min_value=50, max_value=250, value=170, step=1)
    
    with col5:
        weight = st.number_input("Weight (kg)", min_value=20, max_value=300, value=70, step=1)
    
    with col6:
        # Calculate BMI
        bmi = weight / ((height/100) ** 2)
        st.metric("BMI", f"{bmi:.1f}")
        input_data['bmi'] = bmi
    
    st.info(f"BMI Category: {'Underweight' if bmi < 18.5 else 'Normal' if bmi < 25 else 'Overweight' if bmi < 30 else 'Obese'}")

# TAB 2: Lab Results
with tab2:
    st.markdown("### Blood Test Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fasting_glucose = st.number_input(
            "Fasting Glucose (mg/dL)", 
            min_value=50, 
            max_value=400, 
            value=100, 
            step=1,
            help="Normal: <100, Prediabetes: 100-125, Diabetes: ≥126"
        )
        input_data['fasting_glucose_mg_dl'] = fasting_glucose
        
        # Visual indicator
        if fasting_glucose < 100:
            st.success("Normal range")
        elif fasting_glucose < 126:
            st.warning("Prediabetes range")
        else:
            st.error("Diabetes range")
    
    with col2:
        hba1c = st.number_input(
            "HbA1c (%)", 
            min_value=3.0, 
            max_value=15.0, 
            value=5.5, 
            step=0.1,
            help="Normal: <5.7%, Prediabetes: 5.7-6.4%, Diabetes: ≥6.5%"
        )
        input_data['hba1c_percent'] = hba1c
        
        # Visual indicator
        if hba1c < 5.7:
            st.success("Normal range")
        elif hba1c < 6.5:
            st.warning("Prediabetes range")
        else:
            st.error("Diabetes range")
    
    st.markdown("### Cholesterol Levels")
    
    col3, col4 = st.columns(2)
    
    with col3:
        total_cholesterol = st.number_input("Total Cholesterol (mg/dL)", min_value=100, max_value=400, value=200, step=1)
        input_data['total_cholesterol_mg_dl'] = total_cholesterol
        
        ldl = st.number_input("LDL (Bad Cholesterol) (mg/dL)", min_value=50, max_value=300, value=100, step=1)
        input_data['ldl_mg_dl'] = ldl
    
    with col4:
        hdl = st.number_input("HDL (Good Cholesterol) (mg/dL)", min_value=20, max_value=100, value=50, step=1)
        input_data['hdl_mg_dl'] = hdl
        
        triglycerides = st.number_input("Triglycerides (mg/dL)", min_value=50, max_value=500, value=150, step=1)
        input_data['triglycerides_mg_dl'] = triglycerides

# TAB 3: Medical History
with tab3:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Medical Conditions")
        
        family_history = st.selectbox("Family History of Diabetes", options=["No", "Yes"])
        input_data['family_history_diabetes'] = family_history
        
        has_hypertension = st.selectbox("High Blood Pressure (Hypertension)", options=["No", "Yes"])
        input_data['has_hypertension'] = has_hypertension
        
        pcos = st.selectbox("PCOS (Polycystic Ovary Syndrome)", options=["No", "Yes"])
        input_data['pcos'] = pcos
        
        hiv_positive = st.selectbox("HIV Status", options=["Negative", "Positive"])
        input_data['hiv_positive'] = hiv_positive
    
    with col2:
        st.markdown("### Lifestyle")
        
        physically_active = st.selectbox(
            "Physically Active", 
            options=["No", "Yes"],
            help="Regular exercise (at least 30 minutes, 3 times per week)"
        )
        input_data['physically_active'] = physically_active
        
        is_pregnant = st.selectbox("Currently Pregnant", options=["No", "Yes"])
        input_data['is_pregnant'] = is_pregnant

st.markdown("---")

# ============================================================
# PREDICTION BUTTON
# ============================================================
col1, col2, col3 = st.columns([1, 1, 1])

with col2:
    predict_button = st.button("🔍 Predict Diabetes Risk", type="primary", use_container_width=True)

# ============================================================
# MAKE PREDICTION
# ============================================================
if predict_button:
    # Create DataFrame
    input_df = pd.DataFrame([input_data])

    # ===============================
    # 1. Manual binary encoding
    # ===============================
    binary_map = {
        "Yes": 1,
        "No": 0,
        "Positive": 1,
        "Negative": 0
    }

    binary_columns = [
        "family_history_diabetes",
        "has_hypertension",
        "pcos",
        "physically_active",
        "is_pregnant",
        "hiv_positive"
    ]

    for col in binary_columns:
        if col in input_df.columns:
            input_df[col] = input_df[col].map(binary_map)

    # ===============================
    # 2. Label encode non-binary categories
    # ===============================
    categorical_columns = ["sex", "residence"]

    for col in categorical_columns:
        if col in label_encoders:
            input_df[col] = label_encoders[col].transform(input_df[col])

    # ===============================
    # 3. Reorder columns to match training
    # ===============================
    input_df = input_df[feature_names]

    # ===============================
    # 4. Scale features
    # ===============================
    input_scaled = scaler.transform(input_df)

    # ===============================
    # 5. Predict
    # ===============================
    prediction = model.predict(input_scaled)[0]
    prediction_proba = model.predict_proba(input_scaled)[0]

    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    prediction_proba = model.predict_proba(input_scaled)[0]
    
    # Display results
    st.markdown("---")
    st.header("📊 Prediction Results")
    
    # Create columns for result display
    result_col1, result_col2 = st.columns(2)
    
    with result_col1:
        if prediction == 1:
            st.error("### ⚠️ HIGH RISK")
            st.markdown("**Status:** At Risk for Diabetes (Prediabetes or Diabetic)")
            risk_percentage = prediction_proba[1] * 100
            st.metric("Risk Probability", f"{risk_percentage:.1f}%")
        else:
            st.success("### ✅ LOW RISK")
            st.markdown("**Status:** Normal - Low risk for diabetes")
            risk_percentage = prediction_proba[0] * 100
            st.metric("Normal Probability", f"{risk_percentage:.1f}%")
    
    with result_col2:
        # Risk factors identified
        st.markdown("### 🎯 Key Observations")
        
        risk_factors = []
        
        if fasting_glucose >= 100:
            risk_factors.append(f"Fasting glucose elevated ({fasting_glucose} mg/dL)")
        if hba1c >= 5.7:
            risk_factors.append(f"HbA1c elevated ({hba1c}%)")
        if bmi >= 25:
            risk_factors.append(f"BMI indicates overweight/obesity ({bmi:.1f})")
        if family_history == "Yes":
            risk_factors.append("Family history of diabetes")
        if has_hypertension == "Yes":
            risk_factors.append("Has hypertension")
        if physically_active == "No":
            risk_factors.append("Not physically active")
        
        if risk_factors:
            st.warning("**Risk Factors Identified:**")
            for factor in risk_factors:
                st.markdown(f"- {factor}")
        else:
            st.success("No major risk factors identified")
    
    # Recommendations
    st.markdown("---")
    st.header("💡 Recommendations")
    
    if prediction == 1:
        st.markdown("""
        ### Immediate Actions:
        1. **Consult a healthcare professional** for comprehensive evaluation
        2. **Schedule follow-up tests** to confirm diagnosis
        3. **Monitor blood sugar levels** regularly
        
        ### Lifestyle Modifications:
        - Adopt a balanced, low-sugar diet
        - Engage in regular physical activity (30+ minutes daily)
        - Maintain healthy weight
        - Reduce stress levels
        - Get adequate sleep (7-9 hours)
        - Avoid smoking and limit alcohol
        
        ### Medical Follow-up:
        - Schedule appointment with endocrinologist or primary care physician
        - Discuss medication options if necessary
        - Regular monitoring of glucose and HbA1c levels
        """)
    else:
        st.markdown("""
        ### Maintain Healthy Lifestyle:
        1. **Continue healthy habits** to maintain low risk
        2. **Regular health screenings** (annually)
        3. **Stay physically active** and maintain healthy weight
        
        ### Prevention Tips:
        - Eat a balanced diet rich in vegetables, fruits, and whole grains
        - Exercise regularly (at least 150 minutes per week)
        - Maintain healthy BMI (18.5-24.9)
        - Limit sugar and processed foods
        - Stay hydrated
        - Manage stress effectively
        """)
    
    # Disclaimer
    st.markdown("---")
    st.warning("""
    **⚠️ Important Disclaimer:**
    This prediction tool is for screening purposes only and should not be used as a substitute for professional medical advice, 
    diagnosis, or treatment. Always consult with qualified healthcare providers for proper medical evaluation and care.
    """)

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Developed by Olamide Babatunde - Department of Computer Science, University of Ilorin</p>
    <p>Final Year Project - 2025/2026</p>
</div>
""", unsafe_allow_html=True)