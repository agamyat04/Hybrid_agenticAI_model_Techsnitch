from configs.quantization import get_nf4_config
from configs.dora_qlora import get_dora_config
from configs.training_args import get_training_args
from models.model_loader import load_quantized_model
from data.dataset_loader import load_sft_dataset
from train.sft_trainer import run_sft_trainer


def main():
    """
    Entry point for QLoRA + DoRA fine-tuning.

    All model names, datasets, and output paths
    are injected at runtime.
    """

    # ===== PLACEHOLDERS =====
    MODEL_PATH = "<MODEL_NAME_OR_PATH>"
    OUTPUT_DIR = "<OUTPUT_DIR>"

    # To be resolved per model architecture
    TARGET_MODULES = ["<TO_BE_RESOLVED>"]

    # =======================

    quant_config = get_nf4_config()
    lora_config = get_dora_config(TARGET_MODULES)

    model, tokenizer = load_quantized_model(
        MODEL_PATH,
        quant_config,
        lora_config
    )

    # Dataset will be loaded from secure enterprise source
    raw_data = []
    train_dataset = load_sft_dataset(raw_data)

    training_args = get_training_args(OUTPUT_DIR)

    run_sft_trainer(
        model,
        tokenizer,
        train_dataset,
        training_args
    )


if __name__ == "__main__":
    main()
