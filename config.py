import logging

# ==========================================
# SYSTEM LOGGING & CONSTANTS CONFIGURATION
# ==========================================

# Configure logging to output to both console and a log file with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("tahqiq.log", encoding="utf-8")
    ]
)
logger = logging.getLogger("Tahqiq")

# Fixed random state to ensure reproducibility across all experiments
RANDOM_STATE = 42

# Hyperparameter boundaries for Logistic Regression and Vectorizer configuration
MAX_ITER = 1000
MAX_FEATURES = 10000