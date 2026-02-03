from transformers import Trainer

def run_sft_trainer(
    model,
    tokenizer,
    train_dataset,
    training_args
):
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer
    )

    trainer.train()
