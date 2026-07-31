# Implement Zero-shot (Baseline)
def zero_shot_prompt(patient_context):
    """
    Zero-shot prompt with no demonstrations or explicit reasoning framework.
    """

    return f"""
You are analyzing synthetic patient records for an educational healthcare AI project.

Compare the query patient with the three retrieved patients.

Explain:
- the major clinical similarities,
- the important differences,
- and why these patients may have been retrieved as similar.

Do not make diagnoses or treatment recommendations.
Base your answer only on the supplied information.

The similarity retrieval system used the following features:
- TF-IDF representation of recorded clinical conditions
- Age
- Gender
- Encounter count

Race, ethnicity, and medication history were NOT used to calculate
the similarity scores. You may discuss these attributes as observed
similarities or differences, but do not claim that they caused a
patient to be retrieved.

PATIENT RECORDS
----------------
{patient_context}
"""

# Implement Few-shot/In-context learning
def few_shot_prompt(patient_context):
    """
    Few-shot prompt containing examples of the desired comparison behavior.
    """

    return f"""
You are analyzing synthetic patient records for an educational healthcare AI project.

Below are examples showing how patient similarity should be analyzed.

EXAMPLE 1

Query Patient:
Age: 65
Gender: F
Conditions: Hypertension, Diabetes

Similar Patient:
Age: 67
Gender: F
Conditions: Hypertension, Diabetes, Hyperlipidemia

Analysis:
These patients are similar because they are close in age 
and share hypertension and diabetes. The similar patient 
also has hyperlipidemia, which is an important clinical 
difference.

EXAMPLE 2

Query Patient:
Age: 32
Gender: M
Conditions: Asthma, Cough

Similar Patient:
Age: 35
Gender: M
Conditions: Asthma, Fever

Analysis:
Both patients have asthma and are similar in age and gender. 
However, their associated symptoms differ: the query patient 
has cough while the comparison patient has fever.

Now apply the same comparison approach to the following synthetic patient recordds.

Identify:
1. Major clinical similarities
2. Important differences
3. Why the patients may have been retrieved as similar

Do not make diagnoses or treatment recommendations.

The similarity retrieval system used the following features:
- TF-IDF representation of recorded clinical conditions
- Age
- Gender
- Encounter count

Race, ethnicity, and medication history were NOT used to calculate
the similarity scores. You may discuss these attributes as observed
similarities or differences, but do not claim that they caused a
patient to be retrieved.

PATIENT RECORDS
----------------
{patient_context}
"""

# Implement Chain-of-Thought 
def chain_of_thought_prompt(patient_context):
    """
    Chain-of-Thought-style prompt using an explicit, 
    structured comparison process.
    """

    return f"""
You are analyzing synthetic patient records for an educational healthcare AI project.

Analyze the query patient and retrieved patients using the following structured reasoning process.

Step 1: Compare their demographics, ecpecially age and gender.

Step 2: Compare their clinical conditions and identify conditions shared with the query patient.

Step 3: Compare their medication histories and identify
important similarities or differences.

Step 4: Compare healthcare utilization using encounter counts.

Step 5: Based on the evidence above, provide a concise explanation
of why the retrieved patients are clinicalLy similar to the query patient.

Clearly report the result of each step. Use only the
information supplied in the patient records. Do not make
new diagnoses or treatment recommendations.

The similarity retrieval system used the following features:
- TF-IDF representation of recorded clinical conditions
- Age
- Gender
- Encounter count

Race, ethnicity, and medication history were NOT used to calculate
the similarity scores. You may discuss these attributes as observed
similarities or differences, but do not claim that they caused a
patient to be retrieved.

PATIENT RECORDS
----------------
{patient_context}
"""

# Implement Tree-of-Thought
def tree_of_thought_prompt(patient_context):
    """
    Tree-of-Thought-style prompt using multiple
    clinical comparison perspectives.
    """

    return f"""
You are analyzing synthetic patient records for an educational healthcare AI project.

Evaluate the similarity between the query patient and the 
retrieved patients from three separate perspectives.

Perspective A - Clinical Conditions
Compare diagnoses and symptoms. Identify important shared conditions and meaningful differences.

Perspective B - Demographics
Compare age and gender and determine how strongly these characteristics support similarity.

Perspective C - Treatment and Healthcare Utilization
Compare medication histories and encounter counts.

For each perspective:
- identify the strongest evidence for similarity,
- identify important differences,
- and briefly assess how informative that perspective is.

After evaluating all three perspectives, synthesize them 
into a final explanationof why the retrieved patient are 
similar to the query patient.

Base the analysis only on the supplied synthetic records.
Do not make diagnoses or treatment recommendations.

The similarity retrieval system used the following features:
- TF-IDF representation of recorded clinical conditions
- Age
- Gender
- Encounter count

Race, ethnicity, and medication history were NOT used to calculate
the similarity scores. You may discuss these attributes as observed
similarities or differences, but do not claim that they caused a
patient to be retrieved.

PATIENT RECORDS
----------------
{patient_context}
"""

# Perform classification task based on patient similarity
def similarity_classification_prompt(patient_context):
    """
    Few-shot classification prompt combining in-context learning
    with structured evidence evaluation.
    """

    return f"""
You are analyzing synthetic patient records for an
educational healthcare AI project.

Your task is to classify the clinical similarity of each
retrieved patient to the query patient as:

- HIGH
- MODERATE
- LOW

The retrieval system used:
- TF-IDF representation of recorded clinical conditions
- Age
- Gender
- Encounter count

Race, ethnicity, and medication history were NOT used to
calculate retrieval similarity.

Use the following examples as guidance.

EXAMPLE 1

Query:
Age: 60
Gender: F
Conditions: Hypertension, Diabetes
Encounter Count: 8

Candidate:
Age: 62
Gender: F
Conditions: Hypertension, Diabetes
Encounter Count: 7

Classification: HIGH

Reason:
The patients share all recorded conditions, have close ages,
the same gender, and similar encounter counts.


EXAMPLE 2

Query:
Age: 60
Gender: F
Conditions: Hypertension, Diabetes
Encounter Count: 8

Candidate:
Age: 35
Gender: M
Conditions: Asthma
Encounter Count: 2

Classification: LOW

Reason:
The patients have no shared recorded conditions and differ
substantially in age, gender, and healthcare utilization.


Now classify EACH of the three retrieved patients.

For each patient:

1. Evaluate condition overlap.
2. Evaluate age and gender similarity.
3. Evaluate encounter-count similarity.
4. Assign exactly one classification: HIGH, MODERATE, or LOW.
5. Give a brief evidence-based explanation.

Do not use race, ethnicity, or medications as reasons for
the classification.

PATIENT RECORDS
----------------
{patient_context}
"""