# -*- coding: utf-8 -*-
"""pipeline.py — text processing and inference pipeline for Tammini project"""

import numpy as np
import pandas as pd
import re
import torch
import regex as reg
from nltk.tokenize import word_tokenize
from nltk.stem.isri import ISRIStemmer
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

# ----------------- Preload NLP Tools -----------------

arabic_stopwords = set(stopwords.words('arabic'))
stemmer = ISRIStemmer()
Sbert = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v1')
smote = SMOTE(random_state=42)

# ----------------- Text Cleaning -----------------

def clean_text(text):
    cleaned = re.sub(r'['"\n\d,;.،؛.؟]', ' ', text)
    cleaned = re.sub(r'[ؐ-ًؚٟ]*', '', cleaned)
    cleaned = re.sub(r'\s{2,}', ' ', cleaned)

    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols
        u"\U0001F680-\U0001F6FF"  # transport/map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    cleaned = emoji_pattern.sub(r'', cleaned)
    cleaned = re.sub(r'[\u064B-\u0652]', '', cleaned)  # tashkeel removal
    cleaned = re.sub(r'[إأآا]', 'ا', cleaned)
    cleaned = cleaned.replace('ة', 'ه').replace('ى', 'ي').replace("ؤ", "و").replace("ئ", "ي")
    return cleaned.strip()
def clean_and_stem_arabic(text):
    text = re.sub(r'[^\u0600-\u06FF\s]', '', text)
    tokens = text.split()
    filtered = [t for t in tokens if t not in arabic_stopwords]
    stemmed = [stemmer.stem(word) for word in filtered]
    return " ".join(stemmed)

# ----------------- Sentence Embeddings -----------------
def encode_Sbert(questions, answers):
    questions = [clean_and_stem_arabic(clean_text(text)) for text in questions]
    question_embeddings = Sbert.encode(questions, convert_to_tensor=True, normalize_embeddings=True)
    similarities = []
    for _, answer in answers.iterrows():
        answer_embeddings = Sbert.encode(answer.tolist(), convert_to_tensor=True, normalize_embeddings=True)
        row_similarities = cos_sim(question_embeddings, answer_embeddings).diagonal()
        similarities.append(row_similarities.tolist())
    df = pd.DataFrame(similarities, columns=[f"Q{i+1}_sim" for i in range(len(questions))])
    return df

# ----------------- Model Scoring -----------------

def get_score(model, X_test):
    return model.predict_proba(X_test)

# ----------------- Visualization -----------------

def ConfusionMatrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.show()

def LearningCurve(model, X_train, y_train):
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train, cv=5, n_jobs=1, train_sizes=np.linspace(0.1, 1.0, 10)
    )
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Training score', color='blue')
    plt.plot(train_sizes, val_mean, label='Validation score', color='red')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color='blue', alpha=0.2)
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, color='red', alpha=0.2)
    plt.title('Learning Curve')
    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

# ----------------- Model Training -----------------

def RFC(X, y, best_params):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rfc = RandomForestClassifier(**best_params)
    rfc.fit(X_train, y_train)
    y_pred = rfc.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    ConfusionMatrix(y_test, y_pred)
    LearningCurve(rfc, X_train, y_train)
    return rfc, get_score(rfc, X_test)

# ----------------- Upsampling -----------------

def up_sample(X, y):
    return smote.fit_resample(X, y)
