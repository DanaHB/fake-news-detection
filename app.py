import streamlit as st
import joblib
import os
import time
import logging  # 🆕 Added for monitoring hooks

# ==========================================
# 🆕 MONITORING LOGS CONFIGURATION
# ==========================================
# Configures a safe telemetry log file that tracks performance 
# without storing user-submitted news texts or queries.
logging.basicConfig(
    filename='app_monitoring.log',
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)

# ==========================================
# STREAMLIT UI ARCHITECTURE AND CORE SERVING
# ==========================================

# Configure formal window boundaries for the production deployment
st.set_page_config(
    page_title="Tahqiq | Automated Fake News Detection",
    page_icon="🛡️",
    layout="centered"
)

@st.cache_resource
def load_production_model():
    """
    Implements an automatic rollback mechanism (Fallback strategy).
    If the final adapted model fails to load, the system automatically 
    rolls back to the stable baseline model to ensure continuous availability.
    """
    try:
        # Attempt 1: Try loading the latest optimized production model (V2)
        production_tfidf = joblib.load("adapted_tfidf.pkl")
        production_model = joblib.load("adapted_model.pkl")
        return production_tfidf, production_model
    except Exception as e:
        # Rollback Trigger: If V2 fails, log the error and automatically fall back to V1
        st.warning("⚠️ Optimization layer experienced an issue. Automatically rolling back to stable baseline core...")
        logging.error(f"MODEL_LOAD_FAILURE: V2 model failed to load. Reason: {str(e)}") # 🆕 Telemetry hook
        try:
            # Fallback: Load the stable Baseline Model (V1)
            baseline_tfidf = joblib.load("baseline_tfidf.pkl")
            baseline_model = joblib.load("baseline_model.pkl")
            return baseline_tfidf, baseline_model
        except FileNotFoundError:
            # Critical Failure: Both versions are missing
            st.error("❌ Critical System Failure: All model versions are unavailable.")
            logging.critical("SYSTEM_CRASH: Both V1 and V2 models are entirely missing from server deployment.") # 🆕 Telemetry hook
            return None, None

# Initialize cached components seamlessly behind the scenes
tfidf_model, classification_model = load_production_model()

# Header block structuring User-Centered Design metrics
st.title("🛡️ Tahqiq: AI Fake News Detection System")
st.markdown("""
Welcome to **Tahqiq**, an intelligent decision-support platform designed to analyze linguistic patterns and evaluate news credibility in real-time.
This production pipeline operates in absolute compliance with **SDAIA's AI Ethics Principles** (Fairness, Transparency, and Accountability).
""")

st.write("---")

# User textual input area with zero historical logging footprint to preserve user privacy
user_input = st.text_area(
    "Enter the news article text below for verification:", 
    height=220, 
    placeholder="Paste the news title and full content here..."
)

if st.button("Verify Content", type="primary"):
    
    # ⏱️ 1. هنا يبدأ حساب الوقت الإجمالي فوراً عند ضغط الزر
    total_start_time = time.time()
    
    if not user_input.strip():
        st.warning("⚠️ Input buffer is empty. Enter data to execute prediction pipeline.")
    elif classification_model is None:
        st.error("Inference halted. Prediction core engine component is uninitialized.")
        logging.error("INFERENCE_FAILURE: User triggered prediction but classification model is uninitialized.") # 🆕 Telemetry hook
    else:
        with st.spinner("Analyzing structural linguistic distributions..."):
            try:
                # Begin technical performance measurement (Inference Speed Tracking)
                start_time = time.time()
                
                # Import clean_text locally inside the button trigger to avoid circular imports if any
                from data_preprocessing import clean_text
                
                # Step 1: Sanitize input data using the project's standard clean_text pipeline
                cleaned_input = clean_text(user_input)
                
                # Step 2: Extract TF-IDF feature vectors using the optimized vocabulary space
                vectorized_text = tfidf_model.transform([cleaned_input])
                
                # Step 3: Run inference and estimate raw prediction probability states
                prediction = classification_model.predict(vectorized_text)[0]
                probabilities = classification_model.predict_proba(vectorized_text)[0]
                
                # Stop timer immediately after model inference terminates (وقت الموديل الصافي)
                inference_time = time.time() - start_time
                
                # Map prediction class to its exact confidence scale percentage
                confidence = probabilities[prediction] * 100
                
                st.write("### Analysis Results:")
                
                # Step 4: Display color-coded outputs reflecting classification choices (Green=Real, Red=Fake)
                if prediction == 1:
                    st.success("### 🟢 Classified as: REAL NEWS")
                    st.metric(label="Credibility Score (Confidence)", value=f"{confidence:.2f}%")
                    label_str = "REAL"
                else:
                    st.error("### 🔴 Classified as: FAKE NEWS")
                    st.metric(label="Misinformation Score (Confidence)", value=f"{confidence:.2f}%")
                    label_str = "FAKE"
                    
                # ⏱️ 2. هنا ينتهي حساب الوقت الإجمالي بعد رسم الواجهة وإظهار الميتريكس وكل شيء
                total_execution_time = time.time() - total_start_time
                    
                # Log and display non-functional performance speeds to substantiate targets (<1s)
                #st.info(f"⚡ **Internal Inference Time (Model Only):** {inference_time:.4f} seconds")
                #st.success(f"🌐 **Total Execution Time (End-to-End):** {total_execution_time:.4f} seconds (Target: < 1.0s)")

                # ==========================================
                # 🆕 2.5.1 MONITORING HOOK LOGGING TRIGGER
                # ==========================================
                # This explicitly logs performance data into 'app_monitoring.log'.
                # Notice it totally excludes 'user_input' or text strings to ensure absolute privacy!
                logging.info(
                    f"STATUS: Success | INFERENCE_LATENCY: {inference_time:.4f}s | "
                    f"TOTAL_LATENCY: {total_execution_time:.4f}s | "
                    f"PREDICTION: {label_str} | CONFIDENCE: {confidence:.2f}%"
                )

            except Exception as runtime_error:
                # 🆕 Defensive Error Hook: Captures code issues without breaking the web page interface
                total_execution_time = time.time() - total_start_time
                st.error("An error occurred during text engineering analysis. Please check input formatting.")
                logging.error(f"STATUS: Runtime_Error | ERROR_MSG: {str(runtime_error)} | TOTAL_LATENCY: {total_execution_time:.4f}s")

st.write("---")

# Ethical disclaimer ensuring accountability and reinforcing the Human-in-the-loop paradigm
st.caption("""
**Disclaimer & Responsible AI Statement:** Tahqiq is an AI-based decision-support tool. It provides probabilistic classifications based on historical text distributions and linguistic patterns. 
It does not act as an absolute truth engine or a final legal authority. Users are strongly encouraged to maintain critical thinking and independently verify critical information.
""")
