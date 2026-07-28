from pathlib import Path
import pandas as pd

def load_synthea_data(data_dir):
    """Load the required Synthea CSV file."""

    patients = pd.read_csv(data_dir / "patients.csv")
    conditions = pd.read_csv(data_dir / "conditions.csv")
    encounters = pd.read_csv(data_dir / "encounters.csv")
    medications = pd.read_csv(data_dir / "medications.csv")

    return patients, conditions, encounters, medications

# Age calculation
def calculate_age(patients):
    patients = patients.copy()

    patients["BIRTHDATE"] = pd.to_datetime(patients["BIRTHDATE"])
    today = pd.Timestamp.today()

    patients["AGE"] = (
        (today - patients["BIRTHDATE"]).dt.days // 365
    )

    return patients

# Aggregate conditions
def aggregate_conditions(conditions):
    return (
        conditions
        .groupby("PATIENT")["DESCRIPTION"]
        .apply(list)
        .reset_index()
        .rename(columns={"DESCRIPTION": "CONDITIONS"})
    )

# Aggregate medications
def aggregate_medications(medications):
    return (
        medications
        .groupby("PATIENT")["DESCRIPTION"]
        .apply(list)
        .reset_index()
        .rename(columns={"DESCRIPTION": "MEDICATIONS"})
    )

# Encounter counts
def aggregate_encounters(encounters):
    return (
        encounters
        .groupby("PATIENT")
        .size()
        .reset_index(name="ENCOUNTER_COUNT")
    )

# Deduplicate lists
def unique_preserve_order(items):
    return list(dict.fromkeys(items))

# Build master DataFrame
def build_patient_master(
        patients,
        conditions,
        medications,
        encounters,
):
    """
    Build a patient-level master DataFrame from Synthea tables.

    Each row represents one patient and contains demographics,
    healthcare costs, unique conditions, unique medications, and encounter count.
    """

    # Select patient-level demographic and financial features
    patient_demo = patients[
        [
            "Id",
            "AGE",
            "GENDER",
            "RACE",
            "ETHNICITY",
            "HEALTHCARE_EXPENSES",
            "HEALTHCARE_COVERAGE"
        ]
    ].copy()

    # Rename patient identifier so it matches the other Synthea tables
    patient_demo = patient_demo.rename(columns={"Id": "PATIENT"})

    # Aggregate clinical information
    patient_conditions = aggregate_conditions(conditions)
    patient_medications = aggregate_medications(medications)
    encounter_count = aggregate_encounters(encounters)

    # Merge everything into one row per patient
    patient_df = (
        patient_demo
        .merge(patient_conditions, on="PATIENT", how="left")
        .merge(patient_medications, on="PATIENT", how="left")
        .merge(encounter_count, on="PATIENT", how="left")
    )

    # Patients without conditions or medications should have empty lists
    patient_df["CONDITIONS"] = patient_df["CONDITIONS"].apply(
        lambda x: x if isinstance(x, list) else []
    )

    patient_df["MEDICATIONS"] = patient_df["MEDICATIONS"].apply(
        lambda x: x if isinstance(x, list) else []
    )

    # Remove duplicates
    patient_df["CONDITIONS"] = patient_df["CONDITIONS"].apply(
        unique_preserve_order
    )

    patient_df["MEDICATIONS"] = patient_df["MEDICATIONS"].apply(
        unique_preserve_order
    )

    patient_df["ENCOUNTER_COUNT"] = (
        patient_df["ENCOUNTER_COUNT"]
        .fillna(0)
        .astype(int)
    )

    return patient_df

# Save files
def save_patient_master(
    pateint_df,
    processed_path,
):
    pateint_df.to_csv(
        processed_path / "patient_master.csv",
        index=False,
    )

    pateint_df.to_pickle(
        processed_path / "patient_master.pkl"
    )

# Load files
def load_patient_master(processed_path):
    return pd.read_pickle (
        processed_path / "patient_master.pkl"
    )