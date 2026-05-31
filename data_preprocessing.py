import os
import re
import numpy as np
import pandas as pd
#import kagglehub
from config import logger, RANDOM_STATE

def load_isot_dataset():
    import kagglehub
    """
    Downloads and prepares the baseline ISOT Fake News Dataset.
    Combines true and fake components and assigns binary labels (1=Real, 0=Fake).
    """
    try:
        logger.info("Downloading ISOT dataset via kagglehub...")
        path = kagglehub.dataset_download("emineyetm/fake-news-detection-datasets")
        new_path = os.path.join(path, "News _dataset")
        if not os.path.exists(new_path):
            new_path = path
            
        # Load the raw CSV components
        fake = pd.read_csv(os.path.join(new_path, "Fake.csv"))
        true = pd.read_csv(os.path.join(new_path, "True.csv"))
        
        # Inject target labels
        fake["label"] = 0
        true["label"] = 1
        
        # Consolidate and shuffle the dataset using the locked random state
        data = pd.concat([fake, true], axis=0)
        return data.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    except Exception as e:
        logger.error(f"Critical error loading ISOT dataset: {e}")
        raise

def clean_text(text):
    """
    Performs standard sanitization and text cleaning on input strings.
    Removes structural leakages like source agency text, URLs, and non-alphabetic noise.
    """
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    
    text = str(text).lower()
    
    # Text sanitization pipeline matching technical documentation specifications
    text = re.sub(r"http\S+|www\S+", "", text)          # Strip URLs
    text = re.sub(r"\(reuters\)", "", text)             # Remove source agency leakage
    text = re.sub(r"\breuters\b", "", text)             # Remove standard agency names
    text = re.sub(r"\b21st century wire\b", "", text)   # Remove domain leakages
    text = re.sub(r"\bpolitifact\b", "", text)          # Strip external verification signatures
    text = re.sub(r"[^a-zA-Z\s]", "", text)             # Keep only alphabetic sequences
    
    return re.sub(r"\s+", " ", text).strip()            # Normalize whitespaces

def preprocess_dataframe(data):
    """
    Coordinates textual cleaning over a pandas DataFrame.
    Concatenates 'title' and 'text' fields to build full context and drops duplicates.
    """
    data = data.dropna(subset=["text", "title"]).drop_duplicates()
    data["title"] = data["title"].apply(clean_text)
    data["text"] = data["text"].apply(clean_text)
    
    # Feature composition: merging title and body context
    data["content"] = data["title"] + " " + data["text"]
    return data.drop_duplicates(subset=["content"])

def load_welfake():
    """
    Downloads and converts the WELFake evaluation dataset for Domain Adaptation.
    Inverts label definitions to perfectly synchronize target variables across datasets.
    """
    try:
        logger.info("Downloading WELFake dataset via kagglehub...")
        wel_path = kagglehub.dataset_download("saurabhshahane/fake-news-classification")
        wel_df = pd.read_csv(os.path.join(wel_path, "WELFake_Dataset.csv")).dropna(subset=["text"])
        
        wel_df["title"] = wel_df["title"].fillna("")
        wel_df["title"] = wel_df["title"].apply(clean_text)
        wel_df["text"] = wel_df["text"].apply(clean_text)
        wel_df["content"] = wel_df["title"] + " " + wel_df["text"]
        
        # Label inversion to resolve target distribution conflicts across datasets
        wel_df["label"] = 1 - wel_df["label"] 
        return wel_df["content"], wel_df["label"], wel_df
    except Exception as e:
        logger.error(f"Critical error loading WELFake dataset: {e}")
        raise
