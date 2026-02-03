from typing import List, Dict
from data.dataset_loader import load_sft_dataset

REFUSAL_PHRASES = [
    "i cannot",
    "i can't",
    "not allowed",
    "policy",
    "unauthorized",
    "cannot assist",
    "not permitted"
]

def is_refusal(output: str) -> bool:
    output = output.lower()
    return any(p in output for p in REFUSAL_PHRASES)

def is_too_verbose(output: str) -> bool:
    # fallback responses should be short & firm
    return len(output.split()) > 120    

def build_fallback_dataset(r1_data: List[Dict]):
    """
    Filters R1-Distill-SFT for firm, conservative refusals.
    """

    filtered_samples = []

    for sample in r1_data:
        output = sample.get("output", "")

        if not is_refusal(output):
            continue

        if is_too_verbose(output):
            continue

        filtered_samples.append({
            "instruction": sample.get("instruction", "").strip(),
            "input": sample.get("input", "").strip(),
            "output": output.strip()
        })

    return load_sft_dataset(filtered_samples)
