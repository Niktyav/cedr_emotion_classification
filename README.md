# Improving Multi-Label Emotion Classification in Russian Texts

Исследовательский проект по повышению качества multi-label классификации эмоций в русскоязычных текстах на датасете CEDR с использованием Transformer-based моделей  и анализ роли CRF.

## Обзор

Исследование задачи **multi-label классификации эмоций** в русскоязычных текстах на датасете [CEDR](https://huggingface.co/datasets/sagteam/cedr_v1). Проверяются три гипотезы:

| Гипотеза | Результат |
|----------|-----------|
| BERTA fine-tune превзойдёт ELMo-ансамбль из оригинальной статьи | ✅ +0.103 mean Macro F1 |
| CRF-слой улучшит качество за счёт моделирования зависимостей между метками | ❌ −0.01 (объяснение в ноутбуке) |
| BERTA+  Focal Loss | ✅ +0.01 mean Macro F1 относительно  BERTA fine-tune|
| Focal Loss + per-emotion threshold tuning улучшат результат для редких классов | ✅ +0.01(fear) |



## Датасет

[`sagteam/cedr_v1`](https://huggingface.co/datasets/sagteam/cedr_v1) — 9 410 русскоязычных текстов из lj, lenta, twitter с разметкой по 5 эмоциям.

| Эмоция | Индекс | Train |
|--------|--------|-------|
| joy | 0 | ~29% |
| sadness | 1 | ~20% |
| surprise | 2 | ~10% |
| fear | 3 | ~8% |
| anger | 4 | ~5% |
| neutral | — | ~40% |


## Архитектуры моделей


### 1. BERTA + лингвистические признаки + CRF

```
Текст  →  BERTA [CLS] ∈ ℝ⁷⁶⁸
                              ⊕ concat
       →  Лингв. признаки f ∈ ℝ⁴²:
              • Лексические (17): RuSentiLex pos/neg + эмо-словари × 5 классов
              • Психолингвистика (7): местоимения 1/2 лица, отрицания,
                интенсификаторы, каузальные коннекторы, вопрос, восклицание
              • Морфологические (11): POS-распределение, время глагола (pymorphy3)
              • Семантика (7): счётчики по словарям эмоций

       →  [h ; f] ∈ ℝ⁸¹⁰  →  LayerNorm  →  Dropout(0.1)
       →  Linear(810 → 5)  →  logits ∈ ℝ⁵

       →  Document-level CRF:
              score(y) = Σᵢ ψᵢ·yᵢ  +  Σᵢ﹤ⱼ Tᵢⱼ·yᵢ·yⱼ
              2⁵ = 32 конфигурации, точный перебор
       →  argmax P(y | x)  →  5 бинарных предсказаний
```

**Результат**: не улучшает BERTA (Δ = −0.01). Подробный разбор — в ноутбуке.

---

### 2. BERTA+ ← **основная модель** (`src/train.py`)

```
Текст  →  Tokenizer (max_len=128)
       →  sergeyzh/BERTA  (12 слоёв, 768 мерных, заморожен частично)
       →  [CLS] ∈ ℝ⁷⁶⁸
       →  Linear(768 → 256)
       →  LayerNorm(256)
       →  GELU
       →  Dropout(0.3)
       →  Linear(256 → 5)
       →  logits ∈ ℝ⁵

       ↓ при обучении:
       →  Focal Loss  FL(pₜ) = −(1−pₜ)² · log(pₜ),  γ = 2.0
              (снижает вес уверенных предсказаний neutral,
               фокусирует обучение на трудных примерах anger/fear)

       ↓ при инференсе:
       →  Sigmoid × 5
       →  per-emotion threshold:
              joy=0.40, sadness=0.40, surprise=0.45, fear=0.50, anger=0.35
       →  5 бинарных предсказаний
```

**Обучение**: AdamW lr=2e-5, weight_decay=0.01, linear warmup, early stopping patience=2, batch=32, seed=42.  
**Threshold tuning**: сетка 0.20–0.75 с шагом 0.05, оптимизация по val Macro F1 для каждой эмоции отдельно.

---
## Результаты

*Формат метрик: per-emotion Macro F1, как в оригинальной статье (Sboev et al. 2021, Table #2)*

| Модель | joy | sadness | surprise | fear | anger | **mean** |
|--------|-----|---------|----------|------|-------|---------|
| [Статья] Random | 0.44 | 0.44 | 0.41 | 0.39 | 0.39 | 0.41 |
| [Статья] Lexicon | 0.73 | 0.62 | 0.76 | 0.68 | 0.57 | 0.67 |
| [Статья] SVM TF-IDF | 0.67 | 0.71 | 0.67 | 0.66 | 0.50 | 0.64 |
| [Статья] ELMo ensemble | 0.87 | 0.86 | 0.76 | 0.73 | 0.62 | **0.77** |
| TF-IDF + LogReg (наш) | 0.71 | 0.75 | 0.79 | 0.82 | 0.68 | 0.75 |
| BERTA fine-tune | 0.92 | 0.92 | 0.86 | 0.88 | 0.78 | **0.87** |
| BERTA + лингв. + CRF | 0.93 | 0.91 | 0.86 | 0.88 | 0.78 | 0.87 |
| **BERTA+ (FocalLoss + thr)** | **0.93** | **0.92** | **0.86** | **0.89** | **0.79** | **0.88** |
## Основной вывод

ЭксЭксперименты показали, что contextual transformer embeddings уже содержат большую часть психолингвистических сигналов, а handcrafted features и CRF не улучшают качество multilabel emotion classification на данном датасете (≈1.5% multi-label примеров недостаточно для обучения матрицы T).  

## Структура проекта

```text
.
├── notebooks/
│   └── cedr_emotion_classification.ipynb   # Основной ноутбук (все эксперименты)
├── src/
│   ├── train.py                            # Обучение BERTA+
│   └── predict.py                          # Инференс
├── checkpoints/                            # Создаётся после train.py
│   ├── best_model.pt                       # Веса лучшей модели
│   ├── thresholds.json                     # Per-emotion пороги
│   └── tokenizer/                          # Сохранённый токенайзер
├── LICENSE
├── requirements.txt
└── README.md
```

## Установка

```bash
git clone https://github.com/Niktyav/cedr_emotion_classification.git
cd cedr_emotion_classification
pip install -r requirements.txt
```

> **GPU рекомендуется.** Обучение на CPU займёт ~2 ч, на GPU  ~5–15 мин.

---

## Обучение (`train.py`)

Скачивает CEDR v1 с HuggingFace, обучает BERTA+ и сохраняет модель в `checkpoints/`.

**Запуск с параметрами по умолчанию:**
```bash
python src/train.py
```

**Все доступные параметры:**
```bash
python src/train.py \
  --model_name   sergeyzh/BERTA \   # HuggingFace model id
  --output_dir   checkpoints \      # куда сохранять модель
  --epochs       5 \                # максимум эпох (early stopping patience=2)
  --batch_size   32 \               # размер батча
  --max_len      128 \              # максимальная длина токенов
  --lr           2e-5 \             # learning rate (AdamW)
  --weight_decay 0.01 \             # L2 regularization
  --hidden       256 \              # размер скрытого слоя классификатора
  --dropout      0.3 \              # dropout rate
  --focal_gamma  2.0 \              # gamma для Focal Loss
  --patience     2 \                # early stopping patience
  --val_size     0.1 \              # доля валидации из train
  --seed         42
```

**Что происходит при обучении:**
1. Загружается `sagteam/cedr_v1` с HuggingFace (автоматически)
2. Train сплит делится 90/10 на train/val
3. Обучается BERTA+ с Focal Loss (γ=2) и early stopping
4. После обучения выполняется per-emotion threshold tuning на val
5. Финальная оценка на test в формате оригинальной статьи
6. Сохраняется в `checkpoints/`: `best_model.pt`, `thresholds.json`, `tokenizer/`

**Ожидаемый вывод:**
```
Загрузка CEDR v1...
Train: 6 775 | Val: 753 | Test: 1 882

Обучение BERTA+ (5 эпох, patience=2):
  Epoch 01/05 | loss=0.1823 | val Macro F1=0.7912
  Epoch 02/05 | loss=0.1241 | val Macro F1=0.8534
  ...

Threshold tuning на валидации:
  joy       : thr=0.40  val Macro F1=0.9301
  sadness   : thr=0.40  val Macro F1=0.9238
  surprise  : thr=0.45  val Macro F1=0.8599
  fear      : thr=0.50  val Macro F1=0.8880
  anger     : thr=0.35  val Macro F1=0.7930

[BERTA+ (per-emo thr)] Mean Macro F1 = 0.8771
  joy       : 0.9296
  sadness   : 0.9150
  ...

Сохранено:
  Модель:     checkpoints/best_model.pt
  Пороги:     checkpoints/thresholds.json
  Токенайзер: checkpoints/tokenizer/
```

---

## Инференс (`predict.py`)

Загружает обученную модель из `checkpoints/` и предсказывает эмоции.

### Один текст

```bash
python src/predict.py --text "Как же хорошо, когда всё получается!"
```

Вывод:
```
Загрузка модели: sergeyzh/BERTA
  val Macro F1 при сохранении: 0.8771
Пороги: {'joy': 0.4, 'sadness': 0.4, 'surprise': 0.45, 'fear': 0.5, 'anger': 0.35}
Модель готова ✓

Текст:   Как же хорошо, когда всё получается!
Эмоции:  joy
Вероятности:
  ✓ joy       : ████████████████░░░░ 0.823  (thr=0.40)
    sadness   : ██░░░░░░░░░░░░░░░░░░ 0.041  (thr=0.40)
    surprise  : ███░░░░░░░░░░░░░░░░░ 0.112  (thr=0.45)
    fear      : █░░░░░░░░░░░░░░░░░░░ 0.021  (thr=0.50)
    anger     : █░░░░░░░░░░░░░░░░░░░ 0.015  (thr=0.35)
```

### CSV-файл

```bash
python src/predict.py \
  --input_file  texts.csv \         # CSV с колонкой 'text'
  --output_file predictions.csv     # куда сохранить результаты
```

Формат `texts.csv`:
```csv
text
Как же хорошо, когда всё получается!
Мне очень грустно сегодня
Боюсь, что ничего не получится
```

Формат `predictions.csv`:
```csv
text,emotions,prob_joy,prob_sadness,prob_surprise,prob_fear,prob_anger
Как же хорошо...,joy,0.823,0.041,0.112,0.021,0.015
Мне очень грустно...,sadness,0.031,0.891,0.044,0.187,0.022
```

### Интерактивный режим

```bash
python src/predict.py
```

```
Интерактивный режим. Введите текст (или 'exit' для выхода):

>>> Боюсь, что всё пошло не так
Текст:   Боюсь, что всё пошло не так
Эмоции:  fear, sadness

>>> exit
```

### Все параметры predict.py

```bash
python src/predict.py \
  --checkpoint_dir  checkpoints \   # папка с моделью (default: checkpoints)
  --text            "Текст" \       # один текст (опционально)
  --input_file      texts.csv \     # CSV-файл (опционально)
  --output_file     out.csv \       # файл для результатов (опционально)
  --text_column     text \          # название колонки в CSV (default: text)
  --batch_size      32 \            # батч при обработке файла
  --no_probs \                      # не выводить вероятности
  --device          cuda            # cuda | cpu (default: авто)
```

### Использование как библиотеки

```python
from src.predict import EmotionPredictor

predictor = EmotionPredictor("checkpoints")

# Один текст
result = predictor.predict("Мне очень грустно сегодня")
print(result["emotions"])        # ['sadness']
print(result["probabilities"])   # {'joy': 0.03, 'sadness': 0.89, ...}

# Батч текстов
results = predictor.predict_batch(
    ["Как хорошо!", "Боюсь ошибиться"],
    batch_size=32,
    return_probs=True,
)
for r in results:
    print(r["text"], "→", r["emotions"])
```

---

## Технологии

| Библиотека | Версия | Назначение |
|-----------|--------|------------|
| PyTorch | ≥ 2.0 | Обучение моделей |
| Transformers | ≥ 4.35 | BERTA, токенайзер |
| datasets | ≥ 2.14 | Загрузка CEDR |
| pymorphy3 | ≥ 1.0 | Лемматизация русского текста |
| scikit-learn | 1.3.2 | Метрики, TF-IDF |
| umap-learn | 0.5.7 | Визуализация эмбеддингов |
| emoji | ≥ 2.8 | Конвертация эмодзи |



## Направления развития

- contrastive learning
- attention pooling
- ASL/Focal Loss
- dynamic threshold tuning
- emotion-aware augmentation

## Цитирование

```bibtex
@dataset{cedr2021,
  author = {Sboev, Alexander and others},
  title  = {CEDR: Corpus for Emotions Detection in Russian},
  year   = {2021},
  url    = {https://huggingface.co/datasets/sagteam/cedr_v1}
}
```


## License

MIT
