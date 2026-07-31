import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

def call_llm(
        prompt,
        model
):
    """
    Send a prompt to the LLM and return the generated text.
    """

    response = client.responses.create(
        model=model,
        input=prompt
    )

    return response.output_text