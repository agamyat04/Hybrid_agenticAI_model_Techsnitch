from datasets import Dataset
from typing import List, Dict

def validate_sample(sample: Dict) -> bool:
    """
    Enforces enterprise SFT schema.
    Prevents silent dataset corruption.
    """
    required_keys = {"instruction", "input", "output"}

    if not required_keys.issubset(sample.keys()):
        return False

    if not all(isinstance(sample[k], str) for k in required_keys):
        return False

    return True


def format_prompt(sample: Dict) -> str:
    """
    Converts structured enterprise data into a single prompt.
    This is what the model actually sees.
    """

    instruction = sample["instruction"].strip()
    input_text = sample["input"].strip()
    output = sample["output"].strip()

    prompt = f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output}
"""
    return prompt


def load_sft_dataset(
    raw_data: List[Dict]
) -> Dataset:
    """
    Generic SFT dataset loader.
    
    raw_data is expected to be a list of:
    {
        "instruction": str,
        "input": str,
        "output": str
    }
    """

    validated_samples = []

    for sample in raw_data:
        if not validate_sample(sample):
            raise ValueError(f"Invalid sample format: {sample}")

        validated_samples.append({
            "text": format_prompt(sample)
        })

    return Dataset.from_list(validated_samples)
