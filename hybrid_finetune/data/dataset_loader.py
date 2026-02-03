
from datasets import Dataset
from typing import List, Dict

REQUIRED_KEYS = {"instruction", "input", "output"}

def validate_sample(sample: Dict) -> bool:
    """
    Validates that a sample follows the SFT schema.
    """
    if not REQUIRED_KEYS.issubset(sample.keys()):
        return False

    return all(isinstance(sample[k], str) for k in REQUIRED_KEYS)


def format_prompt(sample: Dict) -> str:
    """
    Formats a single SFT prompt.
    """
    return f"""### Instruction:
{sample["instruction"].strip()}

### Input:
{sample["input"].strip()}

### Response:
{sample["output"].strip()}
"""


def load_sft_dataset(samples: List[Dict]) -> Dataset:
    """
    Converts validated SFT samples into a HuggingFace Dataset.
    """
    formatted = []

    for sample in samples:
        if not validate_sample(sample):
            continue

        formatted.append({
            "text": format_prompt(sample)
        })

    return Dataset.from_list(formatted)
