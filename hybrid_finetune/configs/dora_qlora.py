from peft import LoraConfig

def get_dora_config(target_modules):
    """
    target_modules MUST be explicitly provided after inspecting
    the model architecture (e.g. q_proj, v_proj, etc.).

    This prevents silent misconfiguration during fine-tuning.
    """
    if not target_modules or not isinstance(target_modules, list):
        raise ValueError(
            "target_modules must be a non-empty list of module names"
        )

    return LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        use_dora=True,
        target_modules=target_modules,
        # lora_magnitude_vector
    )
