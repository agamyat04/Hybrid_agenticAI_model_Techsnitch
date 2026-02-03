from typing import List, Dict
from data.dataset_loader import load_sft_dataset

# Filtering rules for BASE model
REASONING_KEYWORDS = [
    "explain",
    "why",
    "how",
    "step by step",
    "analyze",
    "compare",
    "describe"
]

REFUSAL_PHRASES = [
    "i cannot",
    "i can't",
    "not allowed",
    "illegal",
    "policy",
    "i am unable"
]

def is_reasoning_task(text: str) -> bool:
    text = text.lower()
    return any(k in text for k in REASONING_KEYWORDS)

def contains_refusal(output: str) -> bool:
    output = output.lower()
    return any(p in output for p in REFUSAL_PHRASES)

# Base dataset builder
def build_base_dataset(openorca_data: List[Dict]):
    """
    Filters OpenOrca for reasoning-heavy, safe samples.
    """

    filtered_samples = []

    for sample in openorca_data:
        instruction = sample.get("instruction", "")
        output = sample.get("output", "")

        if not is_reasoning_task(instruction):
            continue

        if contains_refusal(output):
            continue

        filtered_samples.append({
            "instruction": instruction.strip(),
            "input": sample.get("input", "").strip(),
            "output": output.strip()
        })

    return load_sft_dataset(filtered_samples)
