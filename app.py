import streamlit as st
import joblib
import os
import time
from data_preprocessing import clean_text

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
        try:
            # Fallback: Load the stable Baseline Model (V1)
            baseline_tfidf = joblib.load("baseline_tfidf.pkl")
            baseline_model = joblib.load("baseline_model.pkl")
            return baseline_tfidf, baseline_model
        except FileNotFoundError:
            # Critical Failure: Both versions are missing
            st.error("❌ Critical System Failure: All model versions are unavailable.")
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
    
    # ⏱️ [تعديل] 1. هنا يبدأ حساب الوقت الإجمالي فوراً عند ضغط الزر
    total_start_time = time.time()
    
    if not user_input.strip():
        st.warning("⚠️ Input buffer is empty. Enter data to execute prediction pipeline.")
    elif classification_model is None:
        st.error("Inference halted. Prediction core engine component is uninitialized.")
    else:
        with st.spinner("Analyzing structural linguistic distributions..."):
            # Begin technical performance measurement (Inference Speed Tracking)
            start_time = time.time()
            
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
            else:
                st.error("### 🔴 Classified as: FAKE NEWS")
                st.metric(label="Misinformation Score (Confidence)", value=f"{confidence:.2f}%")
                
            # ⏱️ [تعديل] 2. هنا ينتهي حساب الوقت الإجمالي بعد رسم الواجهة وإظهار الميتريكس وكل شيء
            total_execution_time = time.time() - total_start_time
                
            # Log and display non-functional performance speeds to substantiate targets (<1s)
            st.info(f"⚡ **Internal Inference Time (Model Only):** {inference_time:.4f} seconds")
            st.success(f"🌐 **Total Execution Time (End-to-End):** {total_execution_time:.4f} seconds (Target: < 1.0s)")

st.write("---")

# Ethical disclaimer ensuring accountability and reinforcing the Human-in-the-loop paradigm
st.caption("""
**Disclaimer & Responsible AI Statement:** Tahqiq is an AI-based decision-support tool. It provides probabilistic classifications based on historical text distributions and linguistic patterns. 
It does not act as an absolute truth engine or a final legal authority. Users are strongly encouraged to maintain critical thinking and independently verify critical information.
""")