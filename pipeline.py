# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import re
import torch
from nltk.stem.isri import ISRIStemmer
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import regex as reg

# ----------------- Preprocessing -----------------
arabic_stopwords = set(stopwords.words('arabic'))
stemmer = ISRIStemmer()
Sbert = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v1')

def clean_text(text):
    cleaned = re.sub(r'[\'\"\n\d,;.،؛.؟]', ' ', text)
    cleaned = re.sub(r'[ؐ-ًؚٟ]*', '', cleaned)
    cleaned = re.sub(r'\s{2,}', ' ', cleaned)

    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    cleaned = emoji_pattern.sub(r'', cleaned)
    cleaned = re.sub(r'[\u064B-\u0652]', '', cleaned)
    cleaned = re.sub(r'[إأآا]', 'ا', cleaned)
    cleaned = cleaned.replace('ة', 'ه').replace('ى', 'ي').replace("ؤ", "و").replace("ئ", "ي")
    return cleaned.strip()

def clean_and_stem_arabic(text):
    text = re.sub(r'[^\u0600-\u06FF\s]', '', text)
    tokens = text.split()
    filtered = [t for t in tokens if t not in arabic_stopwords]
    stemmed = [stemmer.stem(word) for word in filtered]
    return " ".join(stemmed)

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

def get_score(model, X_test):
    return model.predict_proba(X_test)

# ----------------- JAIS Translation -----------------
bnb_config = BitsAndBytesConfig(load_in_8bit=True)

tokenizer = AutoTokenizer.from_pretrained("inceptionai/jais-13b-chat", padding_side='left')
model = AutoModelForCausalLM.from_pretrained(
    "inceptionai/jais-13b-chat",
    device_map="auto",
    trust_remote_code=True
)

def translate_to_fusha(text):
    try:
        if not isinstance(text, str) or text.strip() == "":
            return ""
        prompt = f"""حول الجملة التالية من اللهجة العامية إلى اللغة العربية الفصحى:\n"{text}"\nالنتيجة:"""
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=100, do_sample=True, temperature=0.7)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        del inputs, outputs
        torch.cuda.empty_cache()
        return response.split("النتيجة:")[-1].strip()
    except Exception as e:
        print(f"Error translating: {text[:30]}... — {e}")
        torch.cuda.empty_cache()
        return ""
