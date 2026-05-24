# 🛡️ Tahqiq: AI-Powered Fake News Detection

Tahqiq is an end-to-end machine learning pipeline designed to combat misinformation by verifying the authenticity of news articles. The system leverages advanced NLP techniques to classify news as "Real" or "Fake" with high accuracy.

##  Architecture
The project follows a modular, production-ready architecture:
- **Data Pipeline:** Automated downloading and cleaning using `ISOT` and `WELFake` datasets.
- **Model Engine:** Logistic Regression optimized with dynamic Unigram/Bigram feature selection.
- **Web Interface:** Built with Streamlit for real-time inference and monitoring.



##  Features
- **Domain Adaptation:** Training across multiple datasets to ensure model robustness.
- **Performance Monitoring:** Integrated logging system for tracking inference latency and accuracy.
- **Defensive Design:** Automated rollback mechanism if production models fail to load.

##  Quick Start
1. **Clone the repo:**
   `git clone https://github.com/DanaHB/fake-news-detection.git`
2. **Install requirements:**
   `pip install -r requirements.txt`
3. **Launch App:**
   `streamlit run app.py`

## 📊 Performance
The system achieved a robust balance between precision and recall, fine-tuned specifically for news text distribution patterns.
