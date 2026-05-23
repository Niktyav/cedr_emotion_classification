"""
train.py — обучение модели BERTA+ для multi-label классификации эмоций (CEDR)

Архитектура:
    sergeyzh/BERTA  →  [CLS] ∈ ℝ⁷⁶⁸
                     →  Linear(768→256) → LayerNorm → GELU → Dropout(0.3)
                     →  Linear(256→5)   → Focal Loss (γ=2)

После обучения:
    - сохраняет веса модели: checkpoints/best_model.pt
    - сохраняет пороги:      checkpoints/thresholds.json
    - сохраняет токенайзер:  checkpoints/tokenizer/

Использование:
    python train.py
    python train.py --model sergeyzh/BERTA --epochs 5 --lr 2e-5
    python train.py --output_dir my_checkpoints --batch_size 16
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


import argparse
import json
import os
import re
import warnings
from pathlib import Path

import emoji
import nltk
import numpy as np
from datasets import load_dataset
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

warnings.filterwarnings("ignore")
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

import pymorphy3

# ─── Константы ────────────────────────────────────────────────────────────────
LABEL2EMO = {0: "joy", 1: "sadness", 2: "surprise", 3: "fear", 4: "anger"}
EMO2LABEL = {v: k for k, v in LABEL2EMO.items()}
EMOTIONS  = [LABEL2EMO[i] for i in range(5)]
N_EMO     = 5

# ─── Предобработка ────────────────────────────────────────────────────────────
_morph = pymorphy3.MorphAnalyzer()
_stop  = set(stopwords.words("russian"))


def clean_text(text: str) -> str:
    text = re.sub(r"http\S+|@[\w\-]+", "", str(text))
    text = emoji.demojize(text, delimiters=(" ", " ")).replace(":", " ").replace("_", " ")
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\d+", "", text).lower()
    tokens = [
        _morph.parse(w)[0].normal_form
        for w in text.split()
        if w not in _stop and len(w) > 1
    ]
    return " ".join(tokens)


def labels_to_vec(labels_list):
    v = [0] * N_EMO
    for idx in labels_list:
        if 0 <= idx < N_EMO:
            v[idx] = 1
    return v


# ─── Dataset ──────────────────────────────────────────────────────────────────
class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len: int = 128):
        self.texts    = texts
        self.labels   = labels
        self.tok      = tokenizer
        self.max_len  = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tok(
            str(self.texts[idx]),
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].flatten(),
            "attention_mask": enc["attention_mask"].flatten(),
            "labels":         torch.FloatTensor(self.labels[idx]),
        }


# ─── Focal Loss ───────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    Focal Loss для multi-label классификации с сильным дисбалансом.
    FL(p_t) = -(1 - p_t)^gamma * log(p_t)

    gamma=2 фокусирует обучение на трудных примерах (anger, surprise),
    снижая вес уверенных предсказаний majority-класса (neutral).
    """

    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce   = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        proba = torch.sigmoid(logits)
        pt    = torch.where(targets == 1, proba, 1 - proba)
        fl    = ((1 - pt) ** self.gamma) * bce
        return fl.mean()


# ─── Model ────────────────────────────────────────────────────────────────────
class BertaPlusModel(nn.Module):
    """
    BERTA+ — 2-слойный классификатор поверх sergeyzh/BERTA.

    Архитектура:
        BERTA [CLS] ∈ ℝ⁷⁶⁸
          → Linear(768 → hidden)
          → LayerNorm → GELU → Dropout(p)
          → Linear(hidden → K)
          → Focal Loss
    """

    def __init__(
        self,
        model_name: str = "sergeyzh/BERTA",
        n_emotions: int = N_EMO,
        hidden: int = 256,
        dropout: float = 0.3,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.bert      = AutoModel.from_pretrained(model_name)
        hidden_size    = self.bert.config.hidden_size
        self.head      = nn.Sequential(
            nn.Linear(hidden_size, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_emotions),
        )
        self.loss_fn   = FocalLoss(gamma=focal_gamma)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ):
        cls    = self.bert(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state[:, 0]         # [CLS]-токен
        logits = self.head(cls)            # (B, K)

        if labels is not None:
            loss = self.loss_fn(logits, labels)
            return loss, logits
        return logits


# ─── Metrics ──────────────────────────────────────────────────────────────────
def evaluate_paper_format(y_true, y_pred, name: str = "") -> dict:
    """F1 в формате оригинальной статьи: per-emotion binary macro F1."""
    from sklearn.metrics import precision_recall_fscore_support

    macros = []
    for i, emo in enumerate(EMOTIONS):
        _, _, f_mac, _ = precision_recall_fscore_support(
            y_true[:, i], y_pred[:, i], average="macro", zero_division=0
        )
        macros.append(f_mac)
    mean_mac = float(np.mean(macros))
    if name:
        print(f"[{name}] Mean Macro F1 = {mean_mac:.4f}")
        for emo, v in zip(EMOTIONS, macros):
            print(f"  {emo:10s}: {v:.4f}")
    return {f"mac_{e}": v for e, v in zip(EMOTIONS, macros)} | {"mean_macro": mean_mac}


# ─── Per-emotion threshold tuning ─────────────────────────────────────────────
def tune_thresholds(model, val_loader, device, threshold_range=(0.20, 0.75, 0.05)):
    """Подбирает оптимальный порог по val Macro F1 для каждой эмоции."""
    model.eval()
    val_probs, val_true = [], []
    with torch.no_grad():
        for batch in val_loader:
            lg = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
            )
            val_probs.extend(torch.sigmoid(lg).cpu().numpy())
            val_true.extend(batch["labels"].numpy())

    val_probs = np.array(val_probs)
    val_true  = np.array(val_true)
    thresholds_grid = np.arange(*threshold_range)

    best_thresholds = {}
    print("\nThreshold tuning на валидации:")
    for i, emo in enumerate(EMOTIONS):
        best_f1, best_thr = 0.0, 0.5
        for thr in thresholds_grid:
            preds = (val_probs[:, i] > thr).astype(int)
            f = f1_score(val_true[:, i], preds, average="macro", zero_division=0)
            if f > best_f1:
                best_f1, best_thr = f, float(thr)
        best_thresholds[emo] = best_thr
        print(f"  {emo:10s}: thr={best_thr:.2f}  val Macro F1={best_f1:.4f}")

    return best_thresholds


# ─── Data loading ─────────────────────────────────────────────────────────────
def load_cedr(val_size: float = 0.1, seed: int = 42):
    """Загружает CEDR v1 с HuggingFace и выполняет предобработку."""
    print("Загрузка CEDR v1...")
    ds = load_dataset("sagteam/cedr_v1", "main")

    def hf_to_lists(split):
        texts, labels = [], []
        for row in split:
            texts.append(row["text"])
            labels.append(labels_to_vec(row["labels"]))
        return texts, labels

    train_texts, train_labels = hf_to_lists(ds["train"])
    test_texts,  test_labels  = hf_to_lists(ds["test"])

    # val из train
    (tr_texts, val_texts,
     tr_labels, val_labels) = train_test_split(
        train_texts, train_labels,
        test_size=val_size, random_state=seed, shuffle=True
    )

    print(f"Train: {len(tr_texts):,} | Val: {len(val_texts):,} | Test: {len(test_texts):,}")
    return (tr_texts, tr_labels), (val_texts, val_labels), (test_texts, test_labels)


# ─── Main training loop ───────────────────────────────────────────────────────
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model:  {args.model_name}")

    # Данные
    (tr_texts, tr_labels), (val_texts, val_labels), (te_texts, te_labels) = load_cedr(
        val_size=args.val_size, seed=args.seed
    )

    # Токенайзер
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def make_loader(texts, labels, shuffle=False):
        ds = EmotionDataset(texts, labels, tokenizer, args.max_len)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=0)

    tr_loader  = make_loader(tr_texts,  tr_labels,  shuffle=True)
    val_loader = make_loader(val_texts, val_labels)
    te_loader  = make_loader(te_texts,  te_labels)

    # Модель
    model = BertaPlusModel(
        model_name  = args.model_name,
        n_emotions  = N_EMO,
        hidden      = args.hidden,
        dropout     = args.dropout,
        focal_gamma = args.focal_gamma,
    ).to(device)

    # Оптимизатор и scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = len(tr_loader),                # 1 эпоха warmup
        num_training_steps = args.epochs * len(tr_loader),
    )

    # Обучение
    best_mac, best_state, patience_cnt = 0.0, None, 0
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nОбучение BERTA+ ({args.epochs} эпох, patience={args.patience}):")
    for epoch in range(args.epochs):
        # ── train ──
        model.train()
        total_loss = 0.0
        for batch in tr_loader:
            optimizer.zero_grad()
            loss, _ = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["labels"].to(device),
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(tr_loader)

        # ── val ──
        model.eval()
        pv, tv = [], []
        with torch.no_grad():
            for batch in val_loader:
                lg = model(
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                )
                pv.extend(torch.sigmoid(lg).cpu().numpy())
                tv.extend(batch["labels"].numpy())

        pv_bin = (np.array(pv) > 0.5).astype(int)
        mac    = f1_score(np.array(tv), pv_bin, average="macro", zero_division=0)

        print(f"  Epoch {epoch+1:02d}/{args.epochs} | "
              f"loss={avg_loss:.4f} | val Macro F1={mac:.4f}")

        if mac > best_mac:
            best_mac    = mac
            patience_cnt = 0
            best_state  = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f"  Early stopping (patience={args.patience})")
                break

    # ── Восстановить лучшие веса ──
    model.load_state_dict(best_state)
    print(f"\nЛучший val Macro F1 = {best_mac:.4f}")

    # ── Threshold tuning на валидации ──
    best_thresholds = tune_thresholds(model, val_loader, device)

    # ── Финальная оценка на тесте ──
    print("\nОценка на тестовой выборке:")
    model.eval()
    test_probs, test_true = [], []
    with torch.no_grad():
        for batch in te_loader:
            lg = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
            )
            test_probs.extend(torch.sigmoid(lg).cpu().numpy())
            test_true.extend(batch["labels"].numpy())

    test_probs = np.array(test_probs)
    test_true  = np.array(test_true)

    # С порогом 0.5
    preds_05 = (test_probs > 0.5).astype(int)
    evaluate_paper_format(test_true, preds_05, "BERTA+ (thr=0.5)")

    # С per-emotion порогами
    preds_thr = np.zeros_like(preds_05)
    for i, emo in enumerate(EMOTIONS):
        preds_thr[:, i] = (test_probs[:, i] > best_thresholds[emo]).astype(int)
    evaluate_paper_format(test_true, preds_thr, "BERTA+ (per-emo thr)")

    print("\nClassification Report (per-emotion threshold):")
    print(classification_report(test_true, preds_thr, target_names=EMOTIONS, zero_division=0))

    # ── Сохранение ──
    model_path      = output_dir / "best_model.pt"
    thresholds_path = output_dir / "thresholds.json"
    tokenizer_path  = output_dir / "tokenizer"

    torch.save(
        {
            "model_state_dict": best_state,
            "model_name":       args.model_name,
            "hidden":           args.hidden,
            "dropout":          args.dropout,
            "focal_gamma":      args.focal_gamma,
            "val_macro_f1":     best_mac,
            "emotions":         EMOTIONS,
        },
        model_path,
    )
    with open(thresholds_path, "w", encoding="utf-8") as f:
        json.dump(best_thresholds, f, ensure_ascii=False, indent=2)
    tokenizer.save_pretrained(str(tokenizer_path))

    print(f"\nСохранено:")
    print(f"  Модель:     {model_path}")
    print(f"  Пороги:     {thresholds_path}  → {best_thresholds}")
    print(f"  Токенайзер: {tokenizer_path}/")


# ─── CLI ──────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Обучение BERTA+ для multi-label классификации эмоций (CEDR)"
    )
    parser.add_argument("--model_name",   default="sergeyzh/BERTA",
                        help="HuggingFace model id (default: sergeyzh/BERTA)")
    parser.add_argument("--output_dir",   default="checkpoints",
                        help="Папка для сохранения модели (default: checkpoints)")
    parser.add_argument("--epochs",       type=int,   default=5)
    parser.add_argument("--batch_size",   type=int,   default=32)
    parser.add_argument("--max_len",      type=int,   default=128)
    parser.add_argument("--lr",           type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--hidden",       type=int,   default=256,
                        help="Размер скрытого слоя классификатора (default: 256)")
    parser.add_argument("--dropout",      type=float, default=0.3)
    parser.add_argument("--focal_gamma",  type=float, default=2.0,
                        help="Gamma для Focal Loss (default: 2.0)")
    parser.add_argument("--patience",     type=int,   default=2,
                        help="Early stopping patience по val Macro F1")
    parser.add_argument("--val_size",     type=float, default=0.1,
                        help="Доля валидации из train (default: 0.1)")
    parser.add_argument("--seed",         type=int,   default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train(args)
