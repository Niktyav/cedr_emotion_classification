"""
predict.py — инференс BERTA+ для multi-label классификации эмоций

Загружает обученную модель из checkpoints/ и предсказывает эмоции для:
  - одного текста (--text)
  - CSV-файла с текстами (--input_file)
  - интерактивного режима (без аргументов)

Использование:
    # Один текст
    python predict.py --text "Как же хорошо, когда всё получается!"

    # CSV-файл (колонка 'text')
    python predict.py --input_file texts.csv --output_file predictions.csv

    # Интерактивный режим
    python predict.py

    # Другая папка с чекпоинтом
    python predict.py --checkpoint_dir my_checkpoints --text "Боюсь ошибиться"

Формат вывода:
    text | joy | sadness | surprise | fear | anger | emotions
    ─────────────────────────────────────────────────────────
    "Как же хорошо!" | 0.92 | 0.03 | 0.11 | 0.02 | 0.01 | joy
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


import argparse
import json
import re
import sys
import warnings
from pathlib import Path

import emoji
import nltk
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

warnings.filterwarnings("ignore")
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

import pymorphy3

# ─── Константы (должны совпадать с train.py) ──────────────────────────────────
LABEL2EMO = {0: "joy", 1: "sadness", 2: "surprise", 3: "fear", 4: "anger"}
EMOTIONS  = [LABEL2EMO[i] for i in range(5)]
N_EMO     = 5

# ─── Предобработка (та же функция что и в train.py) ───────────────────────────
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


# ─── Model (идентична train.py) ───────────────────────────────────────────────
class BertaPlusModel(nn.Module):
    def __init__(
        self,
        model_name: str = "sergeyzh/BERTA",
        n_emotions: int = N_EMO,
        hidden: int = 256,
        dropout: float = 0.3,
        focal_gamma: float = 2.0,   # не используется при инференсе, но нужен для инициализации
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        H = self.bert.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(H, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_emotions),
        )

    def forward(self, input_ids, attention_mask):
        cls    = self.bert(input_ids=input_ids,
                           attention_mask=attention_mask).last_hidden_state[:, 0]
        return self.head(cls)


# ─── Dataset для батч-инференса ───────────────────────────────────────────────
class InferenceDataset(Dataset):
    def __init__(self, texts: list[str], tokenizer, max_len: int = 128):
        self.texts   = texts
        self.tok     = tokenizer
        self.max_len = max_len

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
        }


# ─── Predictor ────────────────────────────────────────────────────────────────
class EmotionPredictor:
    """
    Обёртка над BERTA+ для удобного инференса.

    Пример:
        predictor = EmotionPredictor("checkpoints")
        result = predictor.predict("Мне очень грустно сегодня")
        print(result["emotions"])       # ['sadness']
        print(result["probabilities"])  # {'joy': 0.03, 'sadness': 0.89, ...}
    """

    def __init__(self, checkpoint_dir: str = "checkpoints", device: str | None = None):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._load()

    def _load(self):
        model_path      = self.checkpoint_dir / "best_model.pt"
        thresholds_path = self.checkpoint_dir / "thresholds.json"
        tokenizer_path  = self.checkpoint_dir / "tokenizer"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Модель не найдена: {model_path}\n"
                "Сначала запустите: python train.py"
            )

        # Загрузка чекпоинта
        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
        model_name  = ckpt.get("model_name",  "sergeyzh/BERTA")
        hidden      = ckpt.get("hidden",      256)
        dropout     = ckpt.get("dropout",     0.3)
        focal_gamma = ckpt.get("focal_gamma", 2.0)

        print(f"Загрузка модели: {model_name}")
        print(f"  val Macro F1 при сохранении: {ckpt.get('val_macro_f1', 'N/A'):.4f}")

        # Токенайзер
        if tokenizer_path.exists():
            self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Модель
        self.model = BertaPlusModel(
            model_name  = model_name,
            n_emotions  = N_EMO,
            hidden      = hidden,
            dropout     = dropout,
            focal_gamma = focal_gamma,
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        # Пороги
        if thresholds_path.exists():
            with open(thresholds_path, encoding="utf-8") as f:
                self.thresholds = json.load(f)
        else:
            print("⚠ thresholds.json не найден, используем 0.5 для всех классов")
            self.thresholds = {emo: 0.5 for emo in EMOTIONS}

        print(f"Пороги: {self.thresholds}")
        print("Модель готова ✓\n")

    @torch.no_grad()
    def _get_probs(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Возвращает сигмоид-вероятности (N, 5)."""
        ds     = InferenceDataset(texts, self.tokenizer)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
        all_probs = []
        for batch in loader:
            logits = self.model(
                batch["input_ids"].to(self.device),
                batch["attention_mask"].to(self.device),
            )
            all_probs.extend(torch.sigmoid(logits).cpu().numpy())
        return np.array(all_probs)

    def predict(self, text: str) -> dict:
        """
        Предсказывает эмоции для одного текста.

        Returns:
            {
                "text":          исходный текст,
                "emotions":      список активных эмоций (может быть пустым → neutral),
                "probabilities": {emotion: probability, ...},
                "thresholds":    применённые пороги,
            }
        """
        probs = self._get_probs([text])[0]
        active = [
            emo for i, emo in enumerate(EMOTIONS)
            if probs[i] > self.thresholds[emo]
        ]
        return {
            "text":          text,
            "emotions":      active if active else ["neutral"],
            "probabilities": {emo: float(round(probs[i], 4))
                              for i, emo in enumerate(EMOTIONS)},
            "thresholds":    self.thresholds,
        }

    def predict_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
        return_probs: bool = False,
    ) -> list[dict]:
        """
        Предсказывает эмоции для списка текстов.

        Args:
            texts:        список текстов
            batch_size:   размер батча
            return_probs: включать ли вероятности в результат

        Returns:
            список словарей {text, emotions, [probabilities]}
        """
        all_probs = self._get_probs(texts, batch_size=batch_size)
        results = []
        for text, probs in zip(texts, all_probs):
            active = [
                emo for i, emo in enumerate(EMOTIONS)
                if probs[i] > self.thresholds[emo]
            ]
            item = {
                "text":     text,
                "emotions": active if active else ["neutral"],
            }
            if return_probs:
                item["probabilities"] = {
                    emo: float(round(probs[i], 4))
                    for i, emo in enumerate(EMOTIONS)
                }
            results.append(item)
        return results


# ─── Форматированный вывод ────────────────────────────────────────────────────
def print_result(result: dict, verbose: bool = True):
    emos  = ", ".join(result["emotions"])
    probs = result.get("probabilities", {})

    print(f"\nТекст:   {result['text'][:80]}")
    print(f"Эмоции:  {emos}")
    if verbose and probs:
        print("Вероятности:")
        for emo in EMOTIONS:
            bar_len = int(probs[emo] * 20)
            bar     = "█" * bar_len + "░" * (20 - bar_len)
            thr     = result.get("thresholds", {}).get(emo, 0.5)
            active  = "✓" if probs[emo] > thr else " "
            print(f"  {active} {emo:10s}: {bar} {probs[emo]:.3f}  (thr={thr:.2f})")


# ─── CLI ──────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Предсказание эмоций с BERTA+"
    )
    parser.add_argument(
        "--checkpoint_dir", default="checkpoints",
        help="Папка с чекпоинтом (default: checkpoints)"
    )
    parser.add_argument(
        "--text", type=str, default=None,
        help="Текст для предсказания"
    )
    parser.add_argument(
        "--input_file", type=str, default=None,
        help="CSV-файл с колонкой 'text'"
    )
    parser.add_argument(
        "--output_file", type=str, default=None,
        help="Куда сохранить результаты (CSV)"
    )
    parser.add_argument(
        "--text_column", default="text",
        help="Название колонки с текстом в CSV (default: text)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32
    )
    parser.add_argument(
        "--no_probs", action="store_true",
        help="Не выводить вероятности"
    )
    parser.add_argument(
        "--device", default=None,
        help="cuda | cpu (по умолчанию: авто)"
    )
    return parser.parse_args()


def main():
    args      = parse_args()
    predictor = EmotionPredictor(
        checkpoint_dir = args.checkpoint_dir,
        device         = args.device,
    )
    verbose = not args.no_probs

    # ── Режим 1: один текст ───────────────────────────────────────────────────
    if args.text:
        result = predictor.predict(args.text)
        print_result(result, verbose=verbose)
        return

    # ── Режим 2: CSV-файл ─────────────────────────────────────────────────────
    if args.input_file:
        import pandas as pd

        df = pd.read_csv(args.input_file)
        if args.text_column not in df.columns:
            print(f"Ошибка: колонка '{args.text_column}' не найдена.")
            print(f"Доступные колонки: {df.columns.tolist()}")
            sys.exit(1)

        texts   = df[args.text_column].fillna("").tolist()
        print(f"Обработка {len(texts):,} текстов...")

        results = predictor.predict_batch(
            texts,
            batch_size   = args.batch_size,
            return_probs = verbose,
        )

        # Добавляем результаты в DataFrame
        df["emotions"] = [", ".join(r["emotions"]) for r in results]
        for emo in EMOTIONS:
            df[f"prob_{emo}"] = [
                r.get("probabilities", {}).get(emo, None) for r in results
            ]

        if args.output_file:
            df.to_csv(args.output_file, index=False, encoding="utf-8")
            print(f"Результаты сохранены: {args.output_file}")
        else:
            print(df[[args.text_column, "emotions"]].to_string(index=False))
        return

    # ── Режим 3: интерактивный ────────────────────────────────────────────────
    print("Интерактивный режим. Введите текст (или 'exit' для выхода):\n")
    while True:
        try:
            text = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nВыход.")
            break

        if not text:
            continue
        if text.lower() in ("exit", "quit", "выход"):
            break

        result = predictor.predict(text)
        print_result(result, verbose=verbose)


if __name__ == "__main__":
    main()
