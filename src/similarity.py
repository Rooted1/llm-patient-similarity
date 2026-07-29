import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix, hstack

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity

# Convert conditions to text
def create_condition_text(patient_df):
    """
    Convert each patient's condition list into a text document for TF-IDF vectorization.
    """
    df = patient_df.copy()

    df["CONDITION_TEXT"] = df["CONDITIONS"].apply(
        lambda conditions: " ".join(conditions)
    )

    return df

# TF-IDF representation
def build_condition_features(patient_df):
    """
    Create TF-IDF features from patient condition description.
    TF-IDF gives higher importance to conditions that help distinguish one patient's profile
    from another while reducing the influence of terms that occur nearly everywhere.
    """

    vectorizer = TfidfVectorizer(
        stop_words="english",
        min_df=2
    )

    condition_matrix = vectorizer.fit_transform(
        patient_df["CONDITION_TEXT"]
    )

    return condition_matrix, vectorizer

# Structured features
def build_structured_features(patient_df):
    """
    Create normalized numeric and encoded categorical patient features.
    """

    numeric_columns = [
        "AGE",
        "ENCOUNTER_COUNT"
    ]

    numeric_data = patient_df[numeric_columns]

    scaler = StandardScaler()
    numeric_scaled = scaler.fit_transform(numeric_data)

    encoder = OneHotEncoder(
        handle_unknown="ignore"
    )

    gender_encoded = encoder.fit_transform(
        patient_df[["GENDER"]]
    )

    structured_matrix = hstack(
        [
            csr_matrix(numeric_scaled),
            gender_encoded
        ]
    )

    return structured_matrix, scaler, encoder

# Combine clinical and structured features
def build_feature_matrix(patient_df):
    """
    Build the complete patient feature matrix using 
    condition TF-IDF and structured patient feature.
    """

    df = create_condition_text(patient_df)

    condition_matrix, vectorizer = build_condition_features(df)

    structured_matrix, scaler, encoder = (
        build_structured_features(df)
    )

    feature_matrix = hstack(
        [
            condition_matrix,
            structured_matrix
        ]
    ).tocsr()

    artifacts = {
        "vectorizer": vectorizer,
        "scaler": scaler,
        "encoder": encoder
    }

    return df, feature_matrix, artifacts

# Calculate similarity
def find_similar_patients(
        patient_df,
        feature_matrix,
        query_index,
        k=3
):
    """
    Retrieve the k most similar patients to a query patient.
    """

    if query_index < 0 or query_index >= len(patient_df):
        raise IndexError("query_index is outside the patient dataset.")

    query_vector = feature_matrix[query_index]

    similarities = cosine_similarity(
        query_vector,
        feature_matrix
    ).flatten()

    # Sort highest similarity first
    ranked_indices = np.argsort(similarities)[::-1]

    # Remove the query patient itself
    ranked_indices = ranked_indices[
        ranked_indices != query_index
    ]

    top_indices = ranked_indices[:k]

    results = patient_df.iloc[top_indices].copy()

    results.insert(
        1,
        "SIMILARITY_SCORE",
        similarities[top_indices]
    )

    return results

# Implement explainable retrieval
def get_shared_conditions(query_patient, matched_patient):
    """
    Return conditions shared by a query patient and a matched patient.
    """

    query_conditions = set(query_patient["CONDITIONS"])
    matched_conditions = set(matched_patient["CONDITIONS"])

    return sorted(query_conditions.intersection(matched_conditions))

def explain_similarity_results(
        patient_df,
        similar_patients,
        query_index,
):
    """
    Create an interpretable summary of retrieve patients, including shared clinical conditions.
    """

    query_patient = patient_df.iloc[query_index]
    rows = []

    for _, patient in similar_patients.iterrows():

        shared_conditions = get_shared_conditions(
            query_patient,
            patient
        )

        rows.append(
            {
                "PATIENT": patient["PATIENT"],
                "SIMILARITY_SCORE": patient["SIMILARITY_SCORE"],
                "AGE": patient["AGE"],
                "GENDER": patient["GENDER"],
                "ENCOUNTER_COUNT": patient["ENCOUNTER_COUNT"],
                "SHARED_CONDITIONS": shared_conditions,
                "CONDITIONS": patient["CONDITIONS"]
            }
        )

    return pd.DataFrame(rows)