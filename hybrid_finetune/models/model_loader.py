from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import prepare_model_for_kbit_training, get_peft_model
from typing import Tuple

def load_quantized_model(
    model_name_or_path: str,
    quant_config,
    dora_config
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
   
    # Tokenizer loading
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=True
    )

    # Ensure padding token exists (important for batching)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model loading in 4-bit NF4
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        quantization_config=quant_config,
        device_map="auto"
    )
    # Prepare model for QLoRA training
    model = prepare_model_for_kbit_training(model)

    # Attach DoRA adapters (weight-decomposed LoRA)
    model = get_peft_model(model, dora_config)
    # Sanity check: print trainable parameters
    model.print_trainable_parameters()

    return model, tokenizer
