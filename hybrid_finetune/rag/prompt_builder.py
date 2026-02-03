
def build_prompt(user_query: str, contexts: list[str]) -> str:
    context_block = "\n\n".join(contexts)

    return f"""
You are an enterprise assistant.
Use the following context only if it is relevant.
Do not invent information.

Context:
{context_block}

User Question:
{user_query}

Answer:
"""
