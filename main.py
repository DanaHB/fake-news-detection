import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Import system utilities and modular processing blocks
from config import logger, RANDOM_STATE
from data_preprocessing import load_isot_dataset, preprocess_dataframe, load_welfake
from model_training import evaluate_and_select_best_vectorizer, tune_hyperparameters, save_artifact

def evaluate_model_metrics(model, tfidf, X_test, y_test, name="Dataset"):
    """
    Evaluates model performance metrics on a specific testing slice.
    Calculates and logs Accuracy and F1-Score for validation traceability.
    """
    X_transformed = tfidf.transform(X_test)
    preds = model.predict(X_transformed)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    logger.info(f"{name} Evaluation Results -> Accuracy: {acc:.4f}, F1-Score: {f1:.4f}")
    return preds

if __name__ == "__main__":
    logger.info("Initializing Tahqiq End-to-End Machine Learning Pipeline...")
    
    # ========================================================
    # STAGE 1: BASELINE DATA PREPARATION & TRAINING (ISOT)
    # ========================================================
    logger.info("--- Starting Stage 1: Baseline Processing ---")
    isot_raw = load_isot_dataset()
    isot_clean = preprocess_dataframe(isot_raw)
    
    # Split ISOT data into 80% Training and 20% Testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        isot_clean["content"], isot_clean["label"], 
        test_size=0.2, random_state=RANDOM_STATE, stratify=isot_clean["label"]
    )
    
    # Dynamically select the best vectorizer and extract text features
    best_tfidf, X_train_best = evaluate_and_select_best_vectorizer(X_train, y_train, "ISOT Stage")
    
    # Tune and execute baseline Logistic Regression classifier training
    best_model = tune_hyperparameters(X_train_best, y_train)
    _ = evaluate_model_metrics(best_model, best_tfidf, X_test, y_test, "ISOT Test")
    
    #  Serialize and save baseline model components to disk
    save_artifact(best_tfidf, "baseline_tfidf.pkl")
    save_artifact(best_model, "baseline_model.pkl")
    
    # ========================================================
    # STAGE 2: ROBUSTNESS & ROBUST CROSS-DOMAIN TESTING
    # ========================================================
    logger.info("--- Starting Stage 2: Cross-Domain Evaluation ---")
    X_wel, y_wel, wel_df = load_welfake()
    
    # Evaluate baseline model performance under Data Shock / Domain Shift conditions
    _ = evaluate_model_metrics(best_model, best_tfidf, X_wel, y_wel, "WELFake Baseline")
    
    # ========================================================
    # STAGE 3: DOMAIN ADAPTATION VIA COMBINED TRAINING
    # ========================================================
    logger.info("--- Starting Stage 3: Domain Adaptation & Optimization ---")
    
    # Isolate 20% of WELFake for combined training and 80% for final blind testing
    X_wel_train, X_wel_test, y_wel_train, y_wel_test = train_test_split(
        wel_df["content"], wel_df["label"], 
        test_size=0.8, random_state=RANDOM_STATE, stratify=wel_df["label"]
    )
    
    # Construct combined training DataFrames from both domain spaces
    isot_train_df = pd.DataFrame({"content": X_train, "label": y_train})
    wel_train_df = pd.DataFrame({"content": X_wel_train, "label": y_wel_train})
    
    # Consolidate and shuffle the final unified cross-domain training set
    combined_train = pd.concat([isot_train_df, wel_train_df], axis=0).sample(
        frac=1, random_state=RANDOM_STATE
    ).reset_index(drop=True)
    
    # Dynamic experimental validation check on the newly integrated training data distribution
    adapted_tfidf, X_comb_best = evaluate_and_select_best_vectorizer(
        combined_train["content"], combined_train["label"], "Domain Adaptation Stage"
    )
    
    # Train the optimized multi-dataset production model configuration
    best_model_adapted = tune_hyperparameters(X_comb_best, combined_train["label"])
    
    # Execute final deployment validation check on isolated unseen test data
    _ = evaluate_model_metrics(best_model_adapted, adapted_tfidf, X_wel_test, y_wel_test, "WELFake Adapted Test")
    
    #  Export production-ready serialized models for web application ingestion
    save_artifact(adapted_tfidf, "adapted_tfidf.pkl")
    save_artifact(best_model_adapted, "adapted_model.pkl")
    
    logger.info("Pipeline execution completed. All production artifacts are successfully locked and stored.")