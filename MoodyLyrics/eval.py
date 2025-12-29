# eval.py
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn
from transformers import BertTokenizer

from dataset import create_data_loader, ID2LABEL
from model import SentimentClassifier


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_preds(model, data_loader, device):
    model.eval()
    preds_all = []
    targets_all = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["targets"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(logits, dim=1)

            preds_all.append(preds.cpu())
            targets_all.append(targets.cpu())

    return torch.cat(preds_all), torch.cat(targets_all)


def plot_confusion_matrix(cm: np.ndarray, labels: list[str], title: str = "Confusion Matrix"):
    import seaborn as sns

    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    ax = sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, default="best_model_state.bin")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_len", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    df = pd.read_csv(args.data_path).dropna()
    df = df[df["Lyrics"] != "Lyrics not found"]

    # Use the same split pattern as training so eval is comparable.
    train_df, temp_df = train_test_split(df, test_size=0.15, random_state=args.seed, stratify=df["Mood"])
    _, test_df = train_test_split(temp_df, test_size=0.5, random_state=args.seed, stratify=temp_df["Mood"])

    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    test_loader = create_data_loader(test_df, tokenizer, args.max_len, args.batch_size)

    n_classes = len(ID2LABEL)
    class_names = [ID2LABEL[i] for i in range(n_classes)]

    model = SentimentClassifier(n_classes).to(device)
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))

    preds, targets = get_preds(model, test_loader, device)

    print("\nClassification report:")
    print(classification_report(targets, preds, target_names=class_names))

    cm = confusion_matrix(targets, preds)
    plot_confusion_matrix(cm, labels=class_names)


if __name__ == "__main__":
    main()
