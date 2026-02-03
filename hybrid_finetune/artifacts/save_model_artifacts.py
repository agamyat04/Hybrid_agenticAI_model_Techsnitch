"""
Step 6: Model Saving

This module freezes the behavior-complete model by saving:
- DoRA / LoRA adapters
- Tokenizer
- Model role metadata
- Offline validation evidence """

import json
from pathlib import Path
from datetime import datetime
def save_model_artifacts(
    model,
    tokenizer,
    model_role_cfg,
    validation_log_path: str,
    output_root: str = "./artifacts"
):
  

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    model_dir = Path(output_root) / model_role_cfg.role / timestamp
    model_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save PEFT adapters (DoRA / LoRA)
    adapter_dir = model_dir / "adapters"
    model.save_pretrained(adapter_dir)

    # 2. Save tokenizer
    tokenizer_dir = model_dir / "tokenizer"
    tokenizer.save_pretrained(tokenizer_dir)

    # 3. Save model metadata (behavioral contract)
    metadata = {
        "model_name": model_role_cfg.name,
        "model_id": model_role_cfg.model_id,
        "role": model_role_cfg.role,
        "responsibility": model_role_cfg.responsibility,
        "behavior_focus": model_role_cfg.behavior_focus,
        "training_type": "QLoRA + DoRA + SFT",
        "offline_validated": True,
        "saved_at_utc": timestamp
    }

    with open(model_dir / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # 4. Save offline validation evidence
    if validation_log_path:
        validation_dest = model_dir / "offline_validation.txt"
        validation_dest.write_text(
            Path(validation_log_path).read_text()
        )

    print(f"\nModel artifacts saved at: {model_dir.resolve()}")
