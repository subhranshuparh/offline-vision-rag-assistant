"""
generate_answer.py
-------------------
Generates a natural-language answer using a LOCAL TinyLlama-1.1B-Chat
model (already in your Hugging Face cache) -- no API, fully offline.
It reasons only over the text metadata of the retrieved results
(filename, predicted class/caption, similarity score), since
TinyLlama cannot see images directly.
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

_tokenizer = None
_model = None


def _load_model():
    global _tokenizer, _model
    if _model is None:
        print(f"Loading local model: {MODEL_NAME} (first call only, then cached)...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype)
        _model.to(device)
        _model.eval()
    return _tokenizer, _model


def _build_context(retrieved_items: list) -> str:
    lines = []
    for i, item in enumerate(retrieved_items, start=1):
        lines.append(
            f"Result {i}: filename={item['filename']}, "
            f"predicted_content={item.get('caption', 'unknown')}, "
            f"similarity_score={item['score']:.3f}"
        )
    return "\n".join(lines)


def generate_answer(user_question: str, retrieved_items: list) -> str:
    tokenizer, model = _load_model()
    device = next(model.parameters()).device

    context = _build_context(retrieved_items)

    system_prompt = (
        "You are a helpful vision analytics assistant. You will be given a list of "
        "retrieved image results (with their predicted content and similarity scores) "
        "and a user question. Answer using ONLY this information. Reference result "
        "numbers when making claims. If the information is not enough to answer "
        "confidently, say so clearly instead of guessing."
    )

    user_prompt = f"Retrieved results:\n{context}\n\nQuestion: {user_question}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.3,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # Strip the prompt portion, keep only the newly generated answer
    answer = full_text[len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):].strip()
    return answer if answer else full_text.strip()


if __name__ == "__main__":
    from retrieval import VisionRetriever

    retriever = VisionRetriever("./index_store")
    results = retriever.query_by_text("dog", k=3)
    answer = generate_answer("What do these results show?", results)
    print(answer)
