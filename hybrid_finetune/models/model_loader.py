from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import prepare_model_for_kbit_training, get_peft_model


def load_quantized_model(
    model_name_or_path: str,
    quant_config,
    lora_config
):
    """
    Loads a model in NF4 quantized mode and attaches DoRA adapters.
    """

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        quantization_config=quant_config,
        device_map="auto"
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    # Helpful for verification
    model.print_trainable_parameters()

    return model, tokenizer
