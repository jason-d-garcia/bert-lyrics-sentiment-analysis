# 🎵 Mood Classification of Song Lyrics with BERT

This repository contains a PyTorch implementation of a **BERT-based classifier** for predicting the **emotional mood of song lyrics**.

The model fine-tunes a pre-trained BERT encoder to classify song lyrics into one of four mood categories derived from **Valence–Arousal theory**.

---

## 📊 Dataset: MoodyLyrics

This project is based on the **MoodyLyrics dataset**, a sentiment-annotated corpus of song lyrics originally collected from `lyrics.wikia.com`.

The dataset is annotated using **Valence and Arousal values** of words (based on **Russell’s circumplex model of affect**) and assigns each song to one of four emotional quadrants:

| Quadrant | Mood     | Valence | Arousal |
|--------|----------|---------|---------|
| Q1     | Happy    | High    | High    |
| Q2     | Angry    | Low     | High    |
| Q3     | Sad      | Low     | Low     |
| Q4     | Relaxed  | High    | Low     |

### Available dataset versions

The original archive contains three versions:

- `ml_raw.xlsx`  
  - 2,595 songs  
  - Full dataset annotated with the four mood categories above

- `ml_balanced.xlsx`  
  - Balanced subset with **500 songs per mood**
  - Total of **2,000 songs**

- `ml_pn_balanced.xlsx`  
  - Same 2,000-song corpus
  - Labels collapsed into **Positive / Negative** sentiment based on Valence only

### Copyright note

All song lyrics were removed from the distributed dataset to comply with copyright regulations.  
Lyrics can be re-downloaded using tools such as:

- https://github.com/tremby/py-lyrics  
- or any comparable lyrics scraping utility

---

## 🧠 Model Architecture

- Base model: `bert-base-cased`
- Input: Full song lyrics (truncated to a fixed maximum length)
- Pooling: BERT `[CLS]` pooled output
- Classifier head:
  - Dropout (p = 0.3)
  - Fully connected linear layer
- Output: 4-class mood prediction

---

## ⚙️ Training Details

- Framework: PyTorch
- Optimizer: AdamW
- Learning rate: 2e-5
- Scheduler: Linear decay (no warmup)
- Loss function: Cross-entropy
- Batch size: 4
- Max sequence length: 160 tokens
- Epochs: 5
- Gradient clipping: Enabled

The model is trained using train/validation/test splits, with the best checkpoint selected based on validation accuracy.

---

## 📁 Expected Input Format

This repository does **not** redistribute lyrics.

The training pipeline expects a CSV file with the following schema:

| Column | Description |
|------|------------|
| Lyrics | Full song lyrics as plain text |
| Mood | One of: `happy`, `sad`, `angry`, `relaxed` |

Example:

Lyrics,Mood  
<LYRICS_TEXT>,sad

---

## 🚀 Usage

Install dependencies:

pip install -r requirements.txt

Train the model:

python train.py --data_path path/to/lyrics.csv

Evaluate a trained model:

python eval.py --checkpoint best_model_state.bin

---

## 📈 Evaluation

Model performance is evaluated using:
- Accuracy
- Precision / Recall / F1-score per class
- Confusion matrix visualization

---

## 🔬 Notes & Motivation

- Lyrics are long-form, metaphor-rich text, making mood classification a challenging NLP task.
- BERT provides strong contextual representations well-suited for modeling emotional language.
- The project focuses on **modeling and evaluation**, not data collection.

---

## 🛠️ Future Work

- Longer-context models (Longformer, BigBird)
- Hierarchical lyric modeling (verse → song)
- Multi-label emotion prediction
- Valence–Arousal regression instead of discrete classes
- Multimodal fusion with audio features

---

## 📜 Citation

If you use the MoodyLyrics dataset, please cite:

Cano, E.; Morisio, M.,  
MoodyLyrics: A Sentiment Annotated Lyrics Dataset,  
ACM Proceedings of the International Conference on Intelligent Systems,  
Metaheuristics & Swarm Intelligence (ISMSI 2017),  
pp. 118–124, Hong Kong, March 2017.  
DOI: 10.1145/3059336.3059340

---

## ⚠️ Disclaimer

Lyrics are copyrighted material and are **not included** in this repository.  
This project is intended for **educational and research purposes only**.
