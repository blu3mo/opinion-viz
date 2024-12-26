# services/embedding.py
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


EMBEDDING_MODEL = "text-embedding-3-small"

def get_embedding(project_name: str, opinion_text: str) -> list:
    """
    OpenAI Embedding API を呼び出して埋め込みベクトル(リスト)を返す
    """
    # "project_name: 意見" というフォーマット
    combined_text = f"{project_name}: {opinion_text}"

    try:
        response = client.embeddings.create(model=EMBEDDING_MODEL,
        input=combined_text)
        emb = response.data[0].embedding  # list of floats
        return emb
    except Exception as e:
        print(f"[WARN] Embedding API error: {e}")
        return []

