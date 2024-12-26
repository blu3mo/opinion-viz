# services/gpt.py
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
import re


def get_opinion_distance(opinion_a: str, opinion_b: str) -> float:
    """
    GPT-4を呼び出して、二つの意見がどれくらい「反対」か(0=全く同じ,1=完全に対立)を返す。
    パース失敗時は 0.5 を返す。
    """
    system_message = {
        "role": "system",
        "content": (
            "You are an assistant that ONLY returns a numeric float from 0 to 1."
            "0 means identical or no difference, 1 means completely opposite."
        )
    }
    user_message = {
        "role": "user",
        "content": f"""
Please read the two opinions and return a single float in [0,1].
Don't add any explanation or text besides the float.

Opinion A: {opinion_a}
Opinion B: {opinion_b}
"""
    }

    try:
        response = client.chat.completions.create(model="gpt-4",
        messages=[system_message, user_message],
        temperature=0.0)
        raw_text = response.choices[0].message.content.strip()

        match = re.search(r"(\d+(\.\d+)?)", raw_text)
        if match:
            val = float(match.group(1))
            val = max(0.0, min(val, 1.0))
            return val
        else:
            return 0.5
    except Exception as e:
        print(f"[WARN] GPT API error: {e}")
        return 0.5
