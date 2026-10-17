"""Data loading, cleaning and encoding for the Heart Disease dashboard."""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).parent / "data" / "heart.csv"

TARGET = "HeartDisease"
CATEGORICAL = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
NUMERIC = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]
# Continuous attributes (FastingBS is a 0/1 flag, so it behaves like a category)
CONTINUOUS = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
# A value of 0 is physiologically impossible for these -> it means "not recorded"
ZERO_MEANS_MISSING = ["RestingBP", "Cholesterol"]

# Human friendly names / descriptions used in the UI
DESCRIPTIONS = {
    "Age": "Age in years",
    "Sex": "Biological sex (M / F)",
    "ChestPainType": "Chest pain type: ASY asymptomatic, ATA atypical angina, NAP non-anginal pain, TA typical angina",
    "RestingBP": "Resting blood pressure (mm Hg)",
    "Cholesterol": "Serum cholesterol (mg/dl)",
    "FastingBS": "Fasting blood sugar > 120 mg/dl (1 = yes)",
    "RestingECG": "Resting ECG: Normal, ST (ST-T abnormality), LVH (left ventricular hypertrophy)",
    "MaxHR": "Maximum heart rate achieved (bpm)",
    "ExerciseAngina": "Exercise-induced angina (Y / N)",
    "Oldpeak": "ST depression induced by exercise vs. rest",
    "ST_Slope": "Slope of the peak-exercise ST segment: Up, Flat, Down",
}


def load_raw(path: Path = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Treat impossible zeros as missing and fill them with the column median."""
    df = df.copy()
    for col in ZERO_MEANS_MISSING:
        df[col] = df[col].replace(0, np.nan)
        df[col] = df[col].fillna(df[col].median())
    return df.drop_duplicates().reset_index(drop=True)


def load_clean(path: Path = DATA_PATH) -> pd.DataFrame:
    return clean(load_raw(path))


def encode(df: pd.DataFrame, columns=None) -> pd.DataFrame:
    """One-hot encode the categorical columns (target excluded).

    If `columns` is given, the result is aligned to it, so a single patient row
    gets exactly the same feature layout the models were trained on.
    """
    X = df.drop(columns=[TARGET], errors="ignore")
    X = pd.get_dummies(X, columns=CATEGORICAL).astype(float)
    if columns is not None:
        X = X.reindex(columns=columns, fill_value=0.0)
    return X
