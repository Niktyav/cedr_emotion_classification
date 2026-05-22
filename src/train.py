import json, os, re, warnings, urllib.request
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import emoji

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import (
    f1_score, precision_recall_fscore_support,
    classification_report, confusion_matrix, hamming_loss
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from transformers import (
    AutoTokenizer, AutoModel,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from datasets import load_dataset

import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

# ─── Воспроизводимость ───────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ─── Константы ───────────────────────────────────────────────────────────────
# CEDR v1: 5 эмоций, индексы 0–4
LABEL2EMO = {0: 'joy', 1: 'sadness', 2: 'surprise', 3: 'fear', 4: 'anger'}
EMO2LABEL = {v: k for k, v in LABEL2EMO.items()}
EMOTIONS  = [LABEL2EMO[i] for i in range(5)]
N_EMO     = 5

MAX_LEN    = 128
BATCH_SIZE = 32
LR         = 2e-5
EPOCHS     = 5
MODEL_NAME = "sergeyzh/BERTA"  

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
print(f'Emotions: {EMOTIONS}')
print(f'Target to beat — ELMo ensemble mean Macro F1: {PAPER["ELMo ensemble"]["mean"]}')
