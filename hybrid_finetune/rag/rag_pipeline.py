from rag.prompt_builder import build_prompt
def rag_inference(
    model,
    tokenizer,
    retriever,
    user_query: str,
    use_rag: bool = True
):
    if use_rag:
        try:
            contexts = retriever.retrieve(user_query)
        except Exception:
            contexts = []
    else:
        contexts = []

    prompt = (
        build_prompt(user_query, contexts)
        if contexts
        else user_query
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False
    )

    return tokenizer.decode(
        output[0], skip_special_tokens=True
    )
