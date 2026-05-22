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