# Implement patient summary structured-to-text for all patients
def generate_patient_summary(
        patient
):
    """
    Convert a patient record into a natural language summary.
    """

    conditions = "\n".join(
        f"- {condition}"
        for condition in patient["CONDITIONS"]
    )
    if patient["MEDICATIONS"]:
        medications = "\n".join(
            f"- {medication}"
            for medication in patient["MEDICATIONS"]
        )
    else:
        medications = "- None recorded"

    return f"""
Patient Profile

Age: {patient['AGE']}
Gender: {patient['GENDER']}
Race: {patient['RACE']}
Ethnicity: {patient['ETHNICITY']}

Encounter Count: {patient['ENCOUNTER_COUNT']}

Conditions:
{conditions}

Medications:
{medications}
""".strip()

# Create summaries for all four patients
def generate_similarity_context(
    query_patient,
    similar_patients,
):
    """
    Create textual context containing the query patient
    and retrieved similar patients.
    """

    context = "QUERY PATIENT\n"
    context += "=" * 50 + "\n"
    context += generate_patient_summary(query_patient)

    for i, (_, patient) in enumerate(
        similar_patients.iterrows(),
        start=1
    ):
        context += f"\n\nSIMILAR PATIENT {i}\n"
        context += "=" * 50 + "\n"
        context += generate_patient_summary(patient)

    return context