import pandas as pd

# Define Evaluation Criteria
EVALUATION_CRITERIA = [
    "FACTUAL_ACCURACY",
    "COMPLETENESS",
    "RETRIEVAL_FAITHFULNESS",
    "CLINICAL_RELEVANCE",
    "CLARITY_ORGANIZATION",
    "EVIDENCE_PRIORITIZATION"
]

# create Evaluation Table
def create_evaluation_table(scores):
    """
    Convert manual prompt-evaluation scores into a DataFrame.

    Parameters
    ----------
    scores : dict
        Dictionary containing one score dictionary per
        prompting strategy.

    Returns
    -------
    pd.DataFrame
        Evaluation table including total and average scores.
    """

    df = pd.DataFrame.from_dict(
        scores,
        orient="index",
    )

    df.index.name = "METHOD"

    df["TOTAL_SCORE"] = df[EVALUATION_CRITERIA].sum(axis=1)

    df["AVERAGE_SCORE"] = (
        df[EVALUATION_CRITERIA].mean(axis=1)
    )

    return df.reset_index()

# Validate evaluation score
def validate_scores(scores):
    """
    Ensure every evaluation score is between 1 and 5.
    """

    for method, method_scores in scores.items():

        for criterion in EVALUATION_CRITERIA:

            if criterion not in method_scores:
                raise ValueError(
                    f"{criterion} missing for {method}"
                )

            score = method_scores[criterion]

            if not 1 <= score <= 5:
                raise ValueError(
                    f"{method}: {criterion} must be between 1 and 5."
                )

    return True

# Define word count
def count_words(text):
    """
    Count the approximate number of words in an LLM response.
    """
    return len(text.split())

# Calculate response length
def calculate_response_lengths(responses):
    """
    Calculate response length for each prompting strategy.
    """

    rows = []

    for method, response in responses.items():
        rows.append(
            {
                "METHOD": method,
                "WORD_COUNT": count_words(response),
            }
        )

    return pd.DataFrame(rows)