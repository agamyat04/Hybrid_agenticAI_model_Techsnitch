from configs.quantization import get_nf4_config
from configs.dora_qlora import get_dora_config
from configs.training_args import get_training_args
from configs.model_roles import MODEL_REGISTRY

from models.model_loader import load_quantized_model
from data.dataset_loader import (
    build_base_dataset,
    build_fallback_dataset
)
from train.sft_trainer import run_sft_trainer


def main():
    """
    Entry point for QLoRA + DoRA fine-tuning.
    """

    # -------------------------------
    # Choose role explicitly
    # -------------------------------
    MODEL_ROLE = "base"  # change to "fallback" for Gemma

    model_cfg = MODEL_REGISTRY[MODEL_ROLE]

    OUTPUT_DIR = f"./outputs/{model_cfg.role}"
    TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

    # -------------------------------
    # Load model
    # -------------------------------
    quant_config = get_nf4_config()
    lora_config = get_dora_config(TARGET_MODULES)

    model, tokenizer = load_quantized_model(
        model_cfg.model_id,
        quant_config,
        lora_config
    )

    # -------------------------------
    # STEP 2: Dataset selection
    # -------------------------------
    if model_cfg.role == "base":
        # OpenOrca data loaded from secure source
        openorca_raw_data = []
        train_dataset = build_base_dataset(openorca_raw_data)
    else:
        # R1-Distill-SFT data loaded from secure source
        r1_raw_data = []
        train_dataset = build_fallback_dataset(r1_raw_data)

    # -------------------------------
    # Training
    # -------------------------------
    training_args = get_training_args(OUTPUT_DIR)

    run_sft_trainer(
        model,
        tokenizer,
        train_dataset,
        training_args
    )


if __name__ == "__main__":
    main()
