# Implement patient summary structured-to-text
def generate_patient_summary(patient):
    """
    Convert a patient record into a natural language summary.
    """

    conditions = "\n".join(
        f"- {condition}"
        for condition in patient["CONDITIONS"]
    )

    medications = "\n".join(
        f"- {medication}"
        for medication in patient["MEDICATIONS"]
    )

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
"""