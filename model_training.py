import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV
from config import logger, MAX_ITER, MAX_FEATURES

def evaluate_and_select_best_vectorizer(X_train, y_train, stage_name="Initial"):
    """
    Dynamic Experimentation Pipeline: Programmatically compares Unigram (1,1) versus
    Bigram (1,2) features using 5-Fold Stratified Cross-Validation on the active split.
    Locks the optimal representation based on empirical mean F1-Scores.
    """
    logger.info(f"[{stage_name}] Initiating dynamic Unigram vs Bigram comparison pipeline...")
    
    # 1. Evaluate pure Unigram configuration
    tfidf_uni = TfidfVectorizer(stop_words="english", max_features=MAX_FEATURES, max_df=0.85, min_df=5, ngram_range=(1, 1))
    X_uni = tfidf_uni.fit_transform(X_train)
    model_uni = LogisticRegression(max_iter=MAX_ITER, class_weight="balanced")
    cv_uni = cross_val_score(model_uni, X_uni, y_train, cv=5, scoring="f1").mean()
    
    # 2. Evaluate combined Bigram configuration
    tfidf_bi = TfidfVectorizer(stop_words="english", max_features=MAX_FEATURES, max_df=0.85, min_df=5, ngram_range=(1, 2))
    X_bi = tfidf_bi.fit_transform(X_train)
    model_bi = LogisticRegression(max_iter=MAX_ITER, class_weight="balanced")
    cv_bi = cross_val_score(model_bi, X_bi, y_train, cv=5, scoring="f1").mean()
    
    logger.info(f"[{stage_name}] Evaluation Summary -> Unigram F1: {cv_uni:.4f} | Bigram F1: {cv_bi:.4f}")
    
    # Programmatic selection based on objective maximization criteria
    if cv_bi > cv_uni:
        logger.info(f"[{stage_name}] Optimization Strategy: Locking Bigram features.")
        return tfidf_bi, X_bi
    else:
        logger.info(f"[{stage_name}] Optimization Strategy: Locking Unigram features.")
        return tfidf_uni, X_uni

def tune_hyperparameters(X_train_best, y_train):
    """
    Executes an exhaustive grid search to find the optimal regularization parameter 'C' 
    for the Logistic Regression model, tracking cross-validated F1 performance.
    """
    param_grid = {"C": [0.1, 1, 10]}
    grid = GridSearchCV(LogisticRegression(max_iter=MAX_ITER, class_weight="balanced"), param_grid, cv=5, scoring="f1")
    grid.fit(X_train_best, y_train)
    logger.info(f"Hyperparameter tuning resolved. Best regularization parameter locked: {grid.best_params_}")
    return grid.best_estimator_

def save_artifact(artifact, filename):
    """
    Serializes operational machine learning artifacts into static binaries (.pkl).
    This supports rapid sub-second initialization within production web serving environments.
    """
    try:
        joblib.dump(artifact, filename)
        logger.info(f"Production artifact successfully exported to: {filename}")
    except Exception as e:
        logger.error(f"Serialization failure for artifact {filename}: {e}")