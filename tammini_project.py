# -*- coding: utf-8 -*-
import streamlit as st
from pymongo import MongoClient
from datetime import datetime
import pandas as pd
import joblib
import os
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import re
import nltk
from nltk.stem.isri import ISRIStemmer
from nltk.corpus import stopwords
import zipfile
if not os.path.exists("sbert_model"):
    with zipfile.ZipFile("Sbert_model.zip", 'r') as zip_ref:
        zip_ref.extractall("sbert_model")
# ----------------- Preprocessing -----------------
nltk.download('stopwords')
arabic_stopwords = set(stopwords.words('arabic'))
stemmer = ISRIStemmer()

# ----------------- Functions -----------------
def clean_text(text):
    cleaned = re.sub(r"[\'\"\n\d,;.،؛.؟]", ' ', text)
    cleaned = re.sub(r"[ؐ-ًؚٟ]*", '', cleaned)
    cleaned = re.sub(r"\s{2,}", ' ', cleaned)
    cleaned = re.sub(r"[\u064B-\u0652]", '', cleaned)
    cleaned = re.sub(r"[إأآا]", 'ا', cleaned)
    cleaned = cleaned.replace('ة','ه').replace('ى','ي').replace('ؤ','و').replace('ئ','ي')
    return cleaned.strip()

def clean_and_stem_arabic(text):
    text = re.sub(r"[^\u0600-\u06FF\s]", '', text)
    tokens = text.split()
    filtered = [t for t in tokens if t not in arabic_stopwords]
    stemmed = [stemmer.stem(word) for word in filtered]
    return ' '.join(stemmed)

def encode_Sbert(questions, answers):
    questions = [clean_and_stem_arabic(clean_text(q)) for q in questions]
    answers = [clean_and_stem_arabic(clean_text(a)) for a in answers]
    q_embeddings = Sbert.encode(questions, convert_to_tensor=True, normalize_embeddings=True)
    a_embeddings = Sbert.encode(answers, convert_to_tensor=True, normalize_embeddings=True)
    similarities = cos_sim(q_embeddings, a_embeddings).diagonal().tolist()
    return pd.DataFrame([similarities], columns=[f"Q{i+1}_sim" for i in range(len(similarities))])

# ----------------- Load Trained Models -----------------
model_path = os.getcwd()
rfc_dep = joblib.load(os.path.join(model_path, 'rfc_dep(1).pkl'))
rfc_anx = joblib.load(os.path.join(model_path, 'rfc_anx(1).pkl'))

DepEncoder = LabelEncoder()
DepEncoder.classes_ = ["Depression", "Healthy"]
AnxEncoder = LabelEncoder()
AnxEncoder.classes_ = ["Anxiety", "Healthy"]


# ----------------- Load SBERT Locally -----------------
Sbert = SentenceTransformer(os.path.join(os.getcwd(), 'sbert_model'))

# ----------------- Database -----------------
uri = "mongodb+srv://tammeni25:mentalhealth255@tamminicluster.nunk6nw.mongodb.net/?retryWrites=true&w=majority&authSource=admin"
client = MongoClient(uri)
db = client["tammini_db"]
users_col = db["users"]
responses_col = db["responses"]

# ----------------- Streamlit Config -----------------
st.set_page_config(page_title="منصة طَمّني", layout="centered")

# ----------------- Auth Pages -----------------
def show_landing_page():
    st.title("منصة طَمّني")
    if st.button("تسجيل الدخول / إنشاء حساب"):
        st.session_state.page = "auth"

def signup():
    st.title("تسجيل حساب جديد")
    username = st.text_input("اسم المستخدم")
    password = st.text_input("كلمة المرور", type="password")
    if st.button("تسجيل"):
        if users_col.find_one({"username": username}):
            st.warning("المستخدم مسجل مسبقاً.")
        else:
            users_col.insert_one({"username": username, "password": password})
            st.success("تم التسجيل بنجاح.")

def login():
    st.title("تسجيل الدخول")
    username = st.text_input("اسم المستخدم")
    password = st.text_input("كلمة المرور", type="password")
    if st.button("دخول"):
        user = users_col.find_one({"username": username, "password": password})
        if user:
            st.session_state.user = username
            st.session_state.page = "questionnaire"
        else:
            st.error("بيانات الدخول غير صحيحة.")

# ----------------- Questionnaire -----------------
def questionnaire():
    st.title("التقييم النفسي")
    gender = st.radio("ما هو جنسك؟", ["ذكر", "أنثى"])
    age = st.radio("ما هي فئتك العمرية؟", ["18-29", "30-39", "40-49", "50+"])

    questions = [
        "هل مررت بفترة استمرت أسبوعين أو أكثر كنت تعاني خلالها من خمسة أعراض أو أكثر؟",
        "هل أدت الأعراض التي مررت بها إلى شعورك بضيق نفسي شديد؟",
        "هل هذه الأعراض لم تكن ناتجة عن تأثير مواد أو حالات طبية؟",
        "هل تعاني من التفكير المفرط أو القلق الزائد تجاه الأمور الحياتية؟",
        "هل تواجه صعوبة في السيطرة على أفكارك القلقة؟",
        "هل يترافق مع التفكير المفرط أو القلق المستمر أعراض جسدية أو نفسية؟"
    ]

    answers = []
    for i, q in enumerate(questions):
        answers.append(st.text_area(f"س{i+1}: {q}"))

    if st.button("تحليل التقييم"):
        if any(a.strip() == '' for a in answers):
            st.error("يرجى الإجابة على جميع الأسئلة.")
            return

        encoded = encode_Sbert(questions, answers)
        dep_pred = rfc_dep.predict(encoded)[0]
        anx_pred = rfc_anx.predict(encoded)[0]

        dep_label = DepEncoder.inverse_transform([dep_pred])[0]
        anx_label = AnxEncoder.inverse_transform([anx_pred])[0]

        if dep_label == "Depression" and anx_label == "Anxiety":
            final_result = "كلا الاكتئاب والقلق"
        elif dep_label == "Depression":
            final_result = "اكتئاب"
        elif anx_label == "Anxiety":
            final_result = "قلق"
        else:
            final_result = "سليم / لا توجد مؤشرات واضحة"

        responses_col.insert_one({
            "username": st.session_state.get("user", "مستخدم مجهول"),
            "gender": gender,
            "age": age,
            **{f"q{i+1}": ans for i, ans in enumerate(answers)},
            "result": final_result,
            "timestamp": datetime.now()
        })

        st.session_state.latest_result = final_result
        st.session_state.page = "result"
        st.experimental_rerun()

# ----------------- Result Page -----------------
def show_results():
    st.title("نتيجة التقييم")
    if "latest_result" in st.session_state:
        st.success(st.session_state.latest_result)
    else:
        st.info("لا توجد نتائج متاحة حالياً.")

# ----------------- Navigation -----------------
if "page" not in st.session_state:
    st.session_state.page = "landing"

if st.session_state.page == "landing":
    show_landing_page()
elif st.session_state.page == "auth":
    auth_option = st.radio("اختر الصفحة", ["تسجيل الدخول", "تسجيل جديد"], horizontal=True)
    if auth_option == "تسجيل الدخول":
        login()
    else:
        signup()
elif st.session_state.page == "questionnaire":
    questionnaire()
elif st.session_state.page == "result":
    show_results()

