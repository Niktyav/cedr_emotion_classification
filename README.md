# Improving Multi-Label Emotion Classification in Russian Texts

Исследовательский проект по повышению качества multi-label классификации эмоций в русскоязычных текстах на датасете CEDR с использованием Transformer-based моделей  и анализ роли CRF.

## Обзор

Исследование задачи **multi-label классификации эмоций** в русскоязычных текстах на датасете [CEDR](https://huggingface.co/datasets/sagteam/cedr_v1). Проверяются три гипотезы:

| Гипотеза | Результат |
|----------|-----------|
| BERTA fine-tune превзойдёт ELMo-ансамбль из оригинальной статьи | ✅ +0.103 mean Macro F1 |
| CRF-слой улучшит качество за счёт моделирования зависимостей между метками | ❌ −0.01 (объяснение в ноутбуке) |
| Focal Loss | ✅ +0.01 mean Macro F1 |
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
| BERTA+ (FocalLoss + thr) | 0.93 | 0.92 | 0.86 | 0.89 | 0.79 | **0.88** |
## Основной вывод

Эксперименты показали, что contextual transformer embeddings уже содержат большую часть психолингвистических сигналов, а handcrafted features и CRF не улучшают качество multilabel emotion classification.

## Структура проекта

```text
.
├── notebooks/
└── src/
```

## Установка

```bash
pip install -r requirements.txt
```

## Запуск обучения

```bash
python src/train.py
```

## Инференс

```bash
python src/predict.py
```

## Технологии

- PyTorch
- Transformers
- scikit-learn
- HuggingFace
- pandas
- numpy

## Направления развития

- contrastive learning
- attention pooling
- ASL/Focal Loss
- dynamic threshold tuning
- emotion-aware augmentation

## License

MIT
