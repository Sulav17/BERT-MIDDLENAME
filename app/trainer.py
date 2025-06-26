import torch
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, BertConfig
from utils.label_map import middle_name_to_id

def train_model():
    print("Loading tokenizer...")
    tokenizer = BertTokenizer.from_pretrained("local-bert", local_files_only=True)
    print("Tokenizer loaded.")

    print("Loading model config...")
    config = BertConfig.from_pretrained(
        "local-bert",
        num_labels=len(middle_name_to_id),
        local_files_only=True
    )

    print("Loading base model with config...")
    model = BertForSequenceClassification.from_pretrained(
        "local-bert",
        config=config,
        local_files_only=True,
        ignore_mismatched_sizes=True  # key part when reinitializing classification head
    )
    print("Model loaded.")

    def preprocess(example):
        enc = tokenizer(
            example["full_name"],
            padding="max_length",
            truncation=True,
            max_length=32
        )
        enc["label"] = middle_name_to_id[example["middle_name"]]
        return enc

    print("Loading dataset...")
    dataset = load_dataset("json", data_files={
        "train": "data/train.json",
        "validation": "data/valid.json"
    })
    print("Dataset loaded.")
    # Filter out examples with labels not in our label map
    print("Filtering dataset to known middle-name labels...")
    dataset = dataset.filter(lambda example: example["middle_name"] in middle_name_to_id)
    # Show new dataset sizes
    train_count = dataset["train"].num_rows if hasattr(dataset["train"], 'num_rows') else len(dataset["train"])
    valid_count = dataset["validation"].num_rows if hasattr(dataset["validation"], 'num_rows') else len(dataset["validation"])
    print(f"Filtered dataset sizes: train={train_count}, validation={valid_count}")

    print("Preprocessing dataset...")
    dataset = dataset.map(preprocess)
    print("Dataset preprocessed.")

    # Determine if CUDA is available and configure accordingly
    use_cuda = torch.cuda.is_available()
    print(f"CUDA available: {use_cuda}. Training on {'GPU' if use_cuda else 'CPU'}.")
    training_args = TrainingArguments(
        output_dir="./models/bert-middle-name",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        logging_dir="./logs",
        logging_steps=10,
        no_cuda=not use_cuda,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
    )

    print("Starting training...")
    trainer.train()
    print("Training finished.")

    print("Saving model and tokenizer...")
    trainer.save_model("./models/bert-middle-name")
    tokenizer.save_pretrained("./models/bert-middle-name")
    print("Saved successfully.")

    return "Training complete!"


if __name__ == "__main__":
    print(train_model())