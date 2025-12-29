# train.py
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from transformers import AdamW, BertTokenizer, get_linear_schedule_with_warmup

from dataset import create_data_loader, ID2LABEL
from model import SentimentClassifier


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples: int):
    model.train()
    losses = []
    correct = 0

    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["targets"].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(logits, targets)

        preds = torch.argmax(logits, dim=1)
        correct += torch.sum(preds == targets).item()
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct / n_examples, float(np.mean(losses, dtype="float32"))


def eval_epoch(model, data_loader, loss_fn, device, n_examples: int):
    model.eval()
    losses = []
    correct = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["targets"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(logits, targets)

            preds = torch.argmax(logits, dim=1)
            correct += torch.sum(preds == targets).item()
            losses.append(loss.item())

    return correct / n_examples, float(np.mean(losses, dtype="float32"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="CSV with columns: Lyrics, Mood")
    parser.add_argument("--checkpoint_path", type=str, default="best_model_state.bin")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_len", type=int, default=160)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = get_device()
    print(f"Using device: {device}")

    df = pd.read_csv(args.data_path).dropna()
    df = df[df["Lyrics"] != "Lyrics not found"]

    # Split
    train_df, temp_df = train_test_split(df, test_size=0.15, random_state=args.seed, stratify=df["Mood"])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=args.seed, stratify=temp_df["Mood"])

    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

    train_loader = create_data_loader(train_df, tokenizer, args.max_len, args.batch_size)
    val_loader = create_data_loader(val_df, tokenizer, args.max_len, args.batch_size)

    n_classes = len(ID2LABEL)
    model = SentimentClassifier(n_classes).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, correct_bias=False)
    total_steps = len(train_loader) * args.epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps,
    )

    loss_fn = nn.CrossEntropyLoss().to(device)

    history = defaultdict(list)
    best_val_acc = 0.0
    ckpt_path = Path(args.checkpoint_path)

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 20)

        train_acc, train_loss = train_epoch(
            model, train_loader, loss_fn, optimizer, device, scheduler, n_examples=len(train_df)
        )
        val_acc, val_loss = eval_epoch(
            model, val_loader, loss_fn, device, n_examples=len(val_df)
        )

        print(f"Train loss {train_loss:.4f}  acc {train_acc:.4f}")
        print(f"Val   loss {val_loss:.4f}  acc {val_acc:.4f}")

        history["train_acc"].append(train_acc)
        history["train_loss"].append(train_loss)
        history["val_acc"].append(val_acc)
        history["val_loss"].append(val_loss)

        if val_acc > best_val_acc:
            torch.save(model.state_dict(), ckpt_path)
            best_val_acc = val_acc
            print(f"Saved new best checkpoint -> {ckpt_path} (val_acc={best_val_acc:.4f})")


if __name__ == "__main__":
    main()
