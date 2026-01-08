# ============================================================
# DUAL-MODE DIABETES PREDICTION SYSTEM - FIXED
# Clinical Model (with lab tests) + Community Model (without)
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
    page_title="Diabetes Risk Prediction - Dual Mode",
    page_icon="🏥",
    layout="wide"
)

# ============================================================
# LOAD MODELS
# ============================================================
@st.cache_resource
def load_clinical_model():
    """Load clinical model (with lab tests)"""
    try:
        model = joblib.load('best_model.pkl')
        scaler = joblib.load('scaler.pkl')
        encoders = joblib.load('label_encoders.pkl')
        features = pd.read_csv('feature_names.csv')['features'].tolist()
        return model, scaler, encoders, features, True
    except FileNotFoundError:
        return None, None, None, None, False

@st.cache_resource
def load_community_model():
    """Load community model (without lab tests)"""
    try:
        model = joblib.load('community_model.pkl')
        scaler = joblib.load('community_scaler.pkl')
        encoders = joblib.load('community_label_encoders.pkl')
        features = pd.read_csv('community_feature_names.csv')['features'].tolist()
        return model, scaler, encoders, features, True
    except FileNotFoundError:
        return None, None, None, None, False

clinical_model, clinical_scaler, clinical_encoders, clinical_features, clinical_available = load_clinical_model()
community_model, community_scaler, community_encoders, community_features, community_available = load_community_model()

# ============================================================
# HEADER
# ============================================================
st.title("🏥 Dual-Mode Diabetes Risk Prediction System")
st.markdown("""
This application offers **two prediction modes** tailored to different healthcare settings:
- **Clinical Mode**: High accuracy with lab test results
- **Community Mode**: Accessible screening without lab tests
""")

st.markdown("---")

# ============================================================
# MODE SELECTION
# ============================================================
st.header("🎯 Select Prediction Mode")

col1, col2 = st.columns(2)

with col1:
    if clinical_available:
        st.success("### 🏥 Clinical Mode")
        st.markdown("""
        **Best for:** Hospitals, Clinics, Diagnostic Centers
        
        **Requires:**
        - Blood test results (Glucose, HbA1c)
        - Cholesterol panel
        - Basic health information
        
        **Accuracy:** ~100%
        """)
        clinical_mode_btn = st.button("Use Clinical Mode", type="primary", use_container_width=True)
    else:
        st.error("Clinical model not available. Please run 2_train_models.py")
        clinical_mode_btn = False

with col2:
    if community_available:
        st.info("### 🏘️ Community Mode")
        st.markdown("""
        **Best for:** Rural areas, Community screening, Home assessment
        
        **Requires:**
        - Age, sex, BMI
        - Medical history
        - Lifestyle information
        
        **Accuracy:** ~91%
        **No lab tests needed!**
        """)
        community_mode_btn = st.button("Use Community Mode", type="secondary", use_container_width=True)
    else:
        st.error("Community model not available. Please run 2b_train_community_model.py")
        community_mode_btn = False

# Initialize session state
if 'mode' not in st.session_state:
    st.session_state.mode = None

if clinical_mode_btn:
    st.session_state.mode = 'clinical'
if community_mode_btn:
    st.session_state.mode = 'community'

st.markdown("---")

# ============================================================
# INPUT FORM
# ============================================================
if st.session_state.mode is not None:
    
    mode = st.session_state.mode
    is_clinical = (mode == 'clinical')
    
    st.header(f"📝 Patient Information - {'Clinical' if is_clinical else 'Community'} Mode")
    
    input_data = {}
    
    # Common inputs for both modes
    tab1, tab2, tab3 = st.tabs(["Demographics", "Lab Results" if is_clinical else "Medical History", "Medical History"])
    
    with tab1:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age (years)", min_value=1, max_value=120, value=30, step=1)
            input_data['age'] = float(age)
        
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
            bmi = weight / ((height/100) ** 2)
            st.metric("BMI", f"{bmi:.1f}")
            input_data['bmi'] = float(bmi)
    
    # Clinical Mode - Lab Results
    if is_clinical:
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
                input_data['fasting_glucose_mg_dl'] = float(fasting_glucose)
                
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
                input_data['hba1c_percent'] = float(hba1c)
                
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
                input_data['total_cholesterol_mg_dl'] = float(total_cholesterol)
                
                ldl = st.number_input("LDL (Bad Cholesterol) (mg/dL)", min_value=50, max_value=300, value=100, step=1)
                input_data['ldl_mg_dl'] = float(ldl)
            
            with col4:
                hdl = st.number_input("HDL (Good Cholesterol) (mg/dL)", min_value=20, max_value=100, value=50, step=1)
                input_data['hdl_mg_dl'] = float(hdl)
                
                triglycerides = st.number_input("Triglycerides (mg/dL)", min_value=50, max_value=500, value=150, step=1)
                input_data['triglycerides_mg_dl'] = float(triglycerides)
    
    # Medical History (both modes)
    with tab3 if is_clinical else tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Medical Conditions")
            
            family_history = st.selectbox("Family History of Diabetes", options=["No", "Yes"])
            input_data['family_history_diabetes'] = family_history
            
            has_hypertension = st.selectbox("High Blood Pressure", options=["No", "Yes"])
            input_data['has_hypertension'] = has_hypertension
            
            pcos = st.selectbox("PCOS", options=["No", "Yes"])
            input_data['pcos'] = pcos
            
            hiv_positive = st.selectbox("HIV Status", options=["Negative", "Positive"])
            input_data['hiv_positive'] = hiv_positive
        
        with col2:
            st.markdown("### Lifestyle")
            
            physically_active = st.selectbox("Physically Active", options=["No", "Yes"])
            input_data['physically_active'] = physically_active
            
            is_pregnant = st.selectbox("Currently Pregnant", options=["No", "Yes"])
            input_data['is_pregnant'] = is_pregnant
    
    st.markdown("---")
    
    # ============================================================
    # PREDICTION
    # ============================================================
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        predict_button = st.button("🔍 Predict Diabetes Risk", type="primary", use_container_width=True)
    
    if predict_button:
        try:
            # Select appropriate model
            if is_clinical:
                model = clinical_model
                scaler = clinical_scaler
                encoders = clinical_encoders
                features = clinical_features
            else:
                model = community_model
                scaler = community_scaler
                encoders = community_encoders
                features = community_features
            
            # Prepare input
            input_df = pd.DataFrame([input_data])
            
            # Encode categorical variables
            for col in input_df.columns:
                if col in encoders:
                    # Encode using the saved encoder
                    input_df[col] = encoders[col].transform(input_df[col])
            
            # Ensure all columns are numeric
            for col in input_df.columns:
                if input_df[col].dtype == 'object':
                    # Manual encoding if still object type
                    input_df[col] = input_df[col].map({'No': 0, 'Yes': 1, 'Male': 0, 'Female': 1, 
                                                       'Urban': 0, 'Rural': 1, 'Negative': 0, 'Positive': 1})
            
            # Ensure correct column order and all features present
            input_df = input_df[features]
            
            # Convert to float
            input_df = input_df.astype(float)
            
            # Scale
            input_scaled = scaler.transform(input_df)
            
            # Predict
            prediction = model.predict(input_scaled)[0]
            prediction_proba = model.predict_proba(input_scaled)[0]
            
            # Display results
            st.markdown("---")
            st.header("📊 Prediction Results")
            
            result_col1, result_col2 = st.columns(2)
            
            with result_col1:
                if prediction == 1:
                    st.error("### ⚠️ HIGH RISK")
                    st.markdown("**Status:** At Risk for Diabetes")
                    risk_percentage = prediction_proba[1] * 100
                    st.metric("Risk Probability", f"{risk_percentage:.1f}%")
                else:
                    st.success("### ✅ LOW RISK")
                    st.markdown("**Status:** Normal - Low diabetes risk")
                    risk_percentage = prediction_proba[0] * 100
                    st.metric("Normal Probability", f"{risk_percentage:.1f}%")
                
                st.info(f"**Mode Used:** {'Clinical' if is_clinical else 'Community'} Model")
            
            with result_col2:
                st.markdown("### 🎯 Key Observations")
                
                risk_factors = []
                
                if is_clinical:
                    if fasting_glucose >= 100:
                        risk_factors.append(f"Elevated glucose ({fasting_glucose} mg/dL)")
                    if hba1c >= 5.7:
                        risk_factors.append(f"Elevated HbA1c ({hba1c}%)")
                
                if bmi >= 25:
                    risk_factors.append(f"Overweight/Obesity (BMI: {bmi:.1f})")
                if family_history == "Yes":
                    risk_factors.append("Family history of diabetes")
                if has_hypertension == "Yes":
                    risk_factors.append("Hypertension present")
                if physically_active == "No":
                    risk_factors.append("Insufficient physical activity")
                
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
                if is_clinical:
                    st.markdown("""
                    ### Immediate Actions (Clinical Mode):
                    1. **Schedule follow-up consultation** with endocrinologist
                    2. **Comprehensive diabetes evaluation** recommended
                    3. **Begin glucose monitoring** as advised
                    
                    ### Treatment Planning:
                    - Discuss medication options if necessary
                    - Create personalized care plan
                    - Regular monitoring schedule
                    """)
                else:
                    st.markdown("""
                    ### Immediate Actions (Community Mode):
                    1. **Visit nearest health facility** for laboratory tests
                    2. **Confirm diagnosis** with proper medical tests
                    3. **Lifestyle modifications** should begin immediately
                    
                    ### Next Steps:
                    - Get fasting glucose and HbA1c tests
                    - Full medical evaluation
                    - Consider switching to Clinical Mode with test results
                    """)
                
                st.markdown("""
                ### Lifestyle Modifications (Both Modes):
                - Adopt low-sugar, balanced diet
                - Exercise 30+ minutes daily
                - Maintain healthy weight
                - Reduce stress
                - Adequate sleep (7-9 hours)
                - Avoid smoking, limit alcohol
                """)
            else:
                st.markdown("""
                ### Maintain Healthy Lifestyle:
                - Continue healthy habits
                - Annual health screenings
                - Stay physically active
                - Balanced nutrition
                """)
            
            st.markdown("---")
            st.warning("""
            **⚠️ Important Disclaimer:**
            This is a screening tool only. Always consult qualified healthcare professionals for diagnosis and treatment.
            """ + (f"\n\n**Note:** Community Mode provides preliminary screening. Laboratory confirmation is recommended for any positive results." if not is_clinical else ""))
        
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")
            st.info("Please check that all fields are filled correctly.")

else:
    st.info("👆 Please select a prediction mode above to begin")

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.header("ℹ️ About the System")
st.sidebar.info("""
**Dual-Mode Diabetes Prediction**

Two specialized models for different settings:

**🏥 Clinical Mode**
- Hospital/clinic use
- Requires lab tests
- ~100% accuracy

**🏘️ Community Mode**
- Rural/home screening
- No lab tests needed
- ~91% accuracy

Final Year Project  
University of Ilorin
""")

st.sidebar.markdown("---")
st.sidebar.header("📊 Model Performance")

if clinical_available and community_available:
    tab1, tab2 = st.sidebar.tabs(["Clinical", "Community"])
    
    with tab1:
        try:
            results = pd.read_csv('model_results.csv')
            st.success(f"**Model:** {results.iloc[0]['Model']}")
            st.metric("Accuracy", f"{results.iloc[0]['Accuracy']*100:.2f}%")
        except:
            st.info("Results not available")
    
    with tab2:
        try:
            results = pd.read_csv('community_model_results.csv')
            st.success(f"**Model:** {results.iloc[0]['Model']}")
            st.metric("Accuracy", f"{results.iloc[0]['Accuracy']*100:.2f}%")
        except:
            st.info("Results not available")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Developed by [Your Name] - Department of Computer Science, University of Ilorin</p>
    <p>Final Year Project - 2024/2025</p>
</div>
""", unsafe_allow_html=True)