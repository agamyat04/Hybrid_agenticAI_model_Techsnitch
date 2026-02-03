from transformers import TrainingArguments

def get_training_args(output_dir: str):
    return TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=3,
    bf16=True,
    logging_steps=10,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,
    evaluation_strategy="no",
    remove_unused_columns=False,
    report_to="none"
    
)

