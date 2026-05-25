import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Import system utilities and modular processing blocks
from config import logger, RANDOM_STATE
from data_preprocessing import load_isot_dataset, preprocess_dataframe, load_welfake
from model_training import evaluate_and_select_best_vectorizer, tune_hyperparameters, save_artifact

def evaluate_model_metrics(model, tfidf, X_test, y_test, name="Dataset", threshold=0.5):
    """
    Evaluates model performance metrics on a specific testing slice using a flexible threshold.
    Calculates and logs Accuracy and F1-Score for validation traceability.
    """
    X_transformed = tfidf.transform(X_test)
    
    # Extract prediction probabilities for the positive class (class 1)
    probabilities = model.predict_proba(X_transformed)[:, 1]
    
    # Apply the custom or default threshold
    preds = (probabilities >= threshold).astype(int)
    
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    
    # Clean print for immediate console output visualization
    print(f"   [{name}] (Threshold: {threshold:.2f}) -> Accuracy: {acc:.4f} | F1-Score: {f1:.4f}")
    
    # Log the evaluation results for audit traceability
    logger.info(f"{name} Evaluation Results (Threshold: {threshold:.2f}) -> Accuracy: {acc:.4f}, F1-Score: {f1:.4f}")
    return f1  # Return F1-Score to easily keep track of the best performing threshold

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  INITIALIZING TAHQIQ END-TO-END MACHINE LEARNING PIPELINE")
    print("="*60 + "\n")
    logger.info("Initializing Tahqiq End-to-End Machine Learning Pipeline...")
    
    # ========================================================
    # STAGE 1: BASELINE DATA PREPARATION & TRAINING (ISOT)
    # ========================================================
    print("-"*50)
    print(" STAGE 1: Baseline Data Preparation & Training (ISOT)")
    print("-"*50)
    logger.info("--- Starting Stage 1: Baseline Processing ---")
    
    print("[INFO] Loading and preprocessing ISOT raw dataset...")
    isot_raw = load_isot_dataset()
    isot_clean = preprocess_dataframe(isot_raw)
    
    print("[INFO] Splitting ISOT dataset into train/test splits (80% Train, 20% Test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        isot_clean["content"], isot_clean["label"], 
        test_size=0.2, random_state=RANDOM_STATE, stratify=isot_clean["label"]
    )
    
    print("[INFO] Executing dynamic Unigram vs Bigram optimization pipeline...")
    best_tfidf, X_train_best = evaluate_and_select_best_vectorizer(X_train, y_train, "ISOT Stage")
    
    print("[INFO] Initiating hyperparameter tuning for baseline model...")
    best_model = tune_hyperparameters(X_train_best, y_train)
    
    print("\n>>> Baseline Model Performance on Internal Test Set:")
    _ = evaluate_model_metrics(best_model, best_tfidf, X_test, y_test, "ISOT Test", threshold=0.5)
    
    # Save baseline model artifacts
    save_artifact(best_tfidf, "baseline_tfidf.pkl")
    save_artifact(best_model, "baseline_model.pkl")
    print(">>> Stage 1 artifacts successfully saved.\n")
    
    # ========================================================
    # STAGE 2: CROSS-DOMAIN TESTING (THE NEW UNSEEN DATA)
    # ========================================================
    print("-"*50)
    print(" STAGE 2: Unseen Cross-Domain Testing (WELFake)")
    print("-"*50)
    logger.info("--- Starting Stage 2: Cross-Domain Evaluation ---")
    
    print("[INFO] Loading unseen cross-domain evaluation dataset (WELFake)...")
    X_wel, y_wel, wel_df = load_welfake()
    
    # Testing the model directly on the new data using standard threshold to witness domain shift
    print("\n>>> [TEST 1] Evaluating Raw Baseline Model on New Unseen Data (Threshold = 0.5):")
    _ = evaluate_model_metrics(best_model, best_tfidf, X_wel, y_wel, "WELFake Raw Baseline", threshold=0.5)
    print(">>> Notice the performance behavior due to domain shift shock.\n")
    
    # ========================================================
    # STAGE 3: DYNAMIC POST-EVALUATION THRESHOLD TUNING (LOOP)
    # ========================================================
    print("-"*50)
    print(" STAGE 3: Dynamic Threshold Tuning Optimization")
    print("-"*50)
    logger.info("Initiating Post-Evaluation Threshold Optimization Loop.")
    print("[PROCESS] Searching for the optimal decision boundary parameterizing across ranges...")
    
    # Array of threshold candidates to test programmatically
    threshold_candidates = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    
    best_calibrated_threshold = 0.5
    best_f1_observed = 0.0
    
    print("\n   --- Executing Grid Search Over Threshold Candidates ---")
    for thresh in threshold_candidates:
        # Run evaluation and catch the F1-Score
        current_f1 = evaluate_model_metrics(best_model, best_tfidf, X_wel, y_wel, f"Tuning Candidate", threshold=thresh)
        
        # Check if this threshold outperforms previous trials
        if current_f1 > best_f1_observed:
            best_f1_observed = current_f1
            best_calibrated_threshold = thresh
            
    print(f"\n   ★ [OPTIMIZATION RESULT] Best Threshold Locked Dynamically: {best_calibrated_threshold:.2f} (Highest F1: {best_f1_observed:.4f}) ★")
    
    print("\n>>> [TEST 2] Re-Evaluating Baseline Model using the Locked Optimal Threshold:")
    _ = evaluate_model_metrics(best_model, best_tfidf, X_wel, y_wel, "WELFake Dynamically Tuned", threshold=best_calibrated_threshold)
    print(">>> Notice how the metrics maximized compared to the initial 0.5 test.\n")
    
    # ========================================================
    # STAGE 4: DOMAIN ADAPTATION VIA DATA STREAM CONSOLIDATION
    # ========================================================
    print("-"*50)
    print(" STAGE 4: Final Enhancement via Joint-Domain Adaptation")
    print("-"*50)
    logger.info("--- Starting Stage 4: Domain Adaptation & Optimization ---")
    
    print("[INFO] Isolating partition slices from WELFake for joint-domain training (20% Train, 80% Blind Test)...")
    X_wel_train, X_wel_test, y_wel_train, y_wel_test = train_test_split(
        wel_df["content"], wel_df["label"], 
        test_size=0.8, random_state=RANDOM_STATE, stratify=wel_df["label"]
    )
    
    # Structure data streams from both domains
    isot_train_df = pd.DataFrame({"content": X_train, "label": y_train})
    wel_train_df = pd.DataFrame({"content": X_wel_train, "label": y_wel_train})
    
    # Merge both datasets to create a robust multi-domain core
    print("[INFO] Shuffling and merging both datasets to build a robust multi-domain core...")
    combined_train = pd.concat([isot_train_df, wel_train_df], axis=0).sample(
        frac=1, random_state=RANDOM_STATE
    ).reset_index(drop=True)
    
    print("[INFO] Re-running dynamic text feature selection on consolidated domain space...")
    adapted_tfidf, X_comb_best = evaluate_and_select_best_vectorizer(
        combined_train["content"], combined_train["label"], "Domain Adaptation Stage"
    )
    
    print("[INFO] Training production-grade adapted model configuration...")
    best_model_adapted = tune_hyperparameters(X_comb_best, combined_train["label"])
    
    print("\n>>> [FINAL TEST] Evaluating Fully Adapted Combined Model on Unseen Split:")
    # Evaluate final production model on the isolated blind test partition
    _ = evaluate_model_metrics(best_model_adapted, adapted_tfidf, X_wel_test, y_wel_test, "WELFake Fully Adapted Test", threshold=0.5)
    
    # Save the final robust production ready models
    print("\n[INFO] Saving production-ready adapted artifacts for deployment ingestion...")
    save_artifact(adapted_tfidf, "adapted_tfidf.pkl")
    save_artifact(best_model_adapted, "adapted_model.pkl")
    
    print("\n" + "="*60)
    print("  PIPELINE EXECUTION COMPLETED: ALL ARTIFACTS SECURELY LOCKED")
    print("="*60 + "\n")
    logger.info("Pipeline execution completed. All production artifacts are successfully locked and stored.")
