# -*- coding: utf-8 -*-
"""
pipeline.py: Utility functions for preprocessing, SBERT encoding, and Arabic translation using JAIS.
Ensure you have set HF_TOKEN in Streamlit secrets or environment.
"""
import os
import re
import torch
import numpy as np
import pandas as pd
import nltk
from nltk.stem.isri import ISRIStemmer
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ----------------- NLTK Setup -----------------
# Download stopwords if needed
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

arabic_stopwords = set(stopwords.words('arabic'))
stemmer = ISRIStemmer()

# ----------------- SBERT Model -----------------
# Load multilingual SBERT with authentication token
Sbert = SentenceTransformer(
    'sentence-transformers/distiluse-base-multilingual-cased-v1',
    use_auth_token=os.environ.get("HF_TOKEN")
)

# ----------------- JAIS Translation Model -----------------
bnb_config = BitsAndBytesConfig(load_in_8bit=True)

tokenizer = AutoTokenizer.from_pretrained(
    "inceptionai/jais-13b-chat",
    padding_side='left',
    use_auth_token=os.environ.get("HF_TOKEN")
)
model = AutoModelForCausalLM.from_pretrained(
    "inceptionai/jais-13b-chat",
    device_map="auto",
    trust_remote_code=True,
    load_in_8bit=True,
    use_auth_token=os.environ.get("HF_TOKEN")
)

# ----------------- Utility Functions -----------------
def clean_text(text: str) -> str:
    """
    Remove punctuation, digits, extra whitespace, emojis, and normalize Arabic letters.
    """
    text = re.sub(r"[\'\"\n\d,;.،؛.؟]", ' ', text)
    text = re.sub(r"[ؐ-ًؚٟ]*", '', text)
    text = re.sub(r"\s{2,}", ' ', text)
    # remove emojis
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub('', text)
    # remove tashkeel
    text = re.sub(r"[\u064B-\u0652]", '', text)
    # normalize letters
    text = re.sub(r"[إأآا]", 'ا', text)
    text = text.replace('ة', 'ه').replace('ى', 'ي').replace('ؤ', 'و').replace('ئ', 'ي')
    return text.strip()


def clean_and_stem_arabic(text: str) -> str:
    """
    Filter non-Arabic characters, remove stopwords, and apply ISRI stemming.
    """
    filtered = []
    for token in re.findall(r"[\u0600-\u06FF]+", text):
        if token not in arabic_stopwords:
            filtered.append(stemmer.stem(token))
    return ' '.join(filtered)


def wrap_answers_in_df(questions: list[str], answers_list: list[str]) -> pd.DataFrame:
    """
    Wrap user input answers (list of strings) into a DataFrame with question keys.
    Example:
        df = wrap_answers_in_df(['س1','س2'], ['جواب1', 'جواب2'])
    returns DataFrame with columns 'س1','س2' and a single row of answers.
    """
    return pd.DataFrame([dict(zip(questions, answers_list))])


def encode_Sbert(questions: list[str], answers: pd.DataFrame) -> pd.DataFrame:
    """
    Encode question-answer similarities via SBERT.
    Returns a DataFrame of shape (n_samples, n_questions) with cosine similarities.
    """
    # preprocess and embed questions
    q_texts = [clean_and_stem_arabic(clean_text(q)) for q in questions]
    q_emb = Sbert.encode(q_texts, convert_to_tensor=True, normalize_embeddings=True)
    rows = []
    for _, row in answers.iterrows():
        ans_texts = [clean_and_stem_arabic(clean_text(row[q])) for q in questions]
        a_emb = Sbert.encode(ans_texts, convert_to_tensor=True, normalize_embeddings=True)
        sims = cos_sim(q_emb, a_emb).diagonal().tolist()
        rows.append(sims)
    cols = [f"Q{i+1}_sim" for i in range(len(questions))]
    return pd.DataFrame(rows, columns=cols)


def get_score(model, X: pd.DataFrame) -> np.ndarray:
    """
    Return model.predict_proba(X).
    """
    return model.predict_proba(X)


def translate_to_fusha(text: str) -> str:
    """
    Translate Arabic dialect to Modern Standard Arabic via JAIS.
    """
    if not text.strip():
        return ''
    prompt = f"حول الجملة التالية من اللهجة العامية إلى اللغة العربية الفصحى:\n\"{text}\"\nالنتيجة:"
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=100, do_sample=True, temperature=0.7)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    torch.cuda.empty_cache()
    return decoded.split('النتيجة:')[-1].strip()


