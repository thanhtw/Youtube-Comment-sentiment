"""
train_and_eval.py
~~~~~~~~~~~~~~~~
Fine-tune a Hugging Face transformer model on YouTube comment data (from YTcomment.xlsx),
evaluate on a test set, and report metrics for research publication.

- Uses 'text' as input and 'label' as ground truth.
- Reports accuracy, precision, recall, F1 (weighted) for test set.
- Saves fine-tuned model and metrics.

Usage:
    python -m analysis.train_and_eval --excel YTcomment.xlsx --output-dir output/
"""


import os
import logging
import argparse
import pandas as pd
import re
import emoji
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
import torch
from datasets import Dataset
def clean_text(text):
    """
    Preprocess text: remove/control special characters, normalize whitespace, handle emojis/icons.
    """
    if not isinstance(text, str):
        return ""
    # Remove control characters
    text = re.sub(r'[\r\n\t]', ' ', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    # Optionally, remove non-printable characters
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    # Convert emojis to text (e.g., 60a -> :smile:)
    text = emoji.demojize(text, delimiters=("", ""))
    # Remove any remaining non-ASCII except CJK (for Chinese)
    # text = re.sub(r'[^\w\s\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff\u3040-\u309f\u30a0-\u30ff:]', '', text)
    return text.strip()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Hugging Face model on YTcomment.xlsx")
    parser.add_argument("--excel", required=True, help="Excel file with 'text' and 'label' columns")
    parser.add_argument("--output-dir", default="output/finetuned_model", help="Directory to save model and metrics")
    parser.add_argument("--model", default="bert-base-chinese", help="Hugging Face model name or path")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set proportion")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Loading data from {args.excel}")

    df = pd.read_excel(args.excel)
    assert "text" in df.columns and "label" in df.columns, "Excel must have 'text' and 'label' columns"
    df = df.dropna(subset=["text", "label"])
    # Preprocess text column
    logger.info("Preprocessing text column (cleaning special characters and icons)...")
    df["text"] = df["text"].apply(clean_text)

    labels = sorted(df["label"].unique())
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    df["label_id"] = df["label"].map(label2id)

    train_df, test_df = train_test_split(df, test_size=args.test_size, random_state=42, stratify=df["label_id"])
    logger.info(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, padding=False)

    train_ds = Dataset.from_pandas(train_df[["text", "label_id"]])
    test_ds = Dataset.from_pandas(test_df[["text", "label_id"]])
    train_ds = train_ds.map(tokenize_fn, batched=True)
    test_ds = test_ds.map(tokenize_fn, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(labels), id2label=id2label, label2id=label2id
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=50,
        report_to=["none"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(-1)
        acc = accuracy_score(labels, preds)
        prec = precision_score(labels, preds, average="weighted", zero_division=0)
        rec = recall_score(labels, preds, average="weighted", zero_division=0)
        f1 = f1_score(labels, preds, average="weighted", zero_division=0)
        return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    logger.info("Starting training...")
    trainer.train()
    logger.info("Training complete. Evaluating on test set...")
    metrics = trainer.evaluate()
    logger.info(f"Test metrics: {metrics}")

    # Save metrics and classification report
    preds = trainer.predict(test_ds)
    y_true = preds.label_ids
    y_pred = preds.predictions.argmax(-1)
    report = classification_report(y_true, y_pred, target_names=[id2label[i] for i in range(len(labels))], zero_division=0)
    with open(os.path.join(args.output_dir, "test_metrics.txt"), "w", encoding="utf-8") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
        f.write("\nClassification Report:\n")
        f.write(report)
    logger.info("Saved metrics and classification report.")
    trainer.save_model(args.output_dir)
    logger.info(f"Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()
