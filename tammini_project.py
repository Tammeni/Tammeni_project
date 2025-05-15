import streamlit as st
from pymongo import MongoClient
from datetime import datetime
import pandas as pd
import joblib
import re
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from nltk.corpus import stopwords
from nltk.stem.isri import ISRIStemmer
import nltk

# ----------------- NLTK Setup -----------------
nltk.download('stopwords')
arabic_stopwords = set(stopwords.words('arabic'))
stemmer = ISRIStemmer()
# ----------------- MongoDB Connection -----------------
uri = "mongodb+srv://tammeni25:mentalhealth255@tamminicluster.nunk6nw.mongodb.net/?retryWrites=true&w=majority&authSource=admin"
client = MongoClient(uri)
db = client["tammini_db"]
users_col = db["users"]
responses_col = db["responses"]
# ----------------- Load Trained Models -----------------
rfc_dep = joblib.load("rfc_dep.pkl")
rfc_anx = joblib.load("rfc_anx.pkl")
DepEncoder = {0: "Healthy", 1: "Depression"}
AnxEncoder = {0: "Healthy", 1: "Anxiety"}
# ----------------- SBERT Model -----------------
Sbert = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v1')

# ----------------- Preprocessing Functions -----------------
def clean_text(text):
    text = re.sub(r"[\'\"\n\d,;.،؛.؟]", ' ', text)
    text = re.sub(r"[ؐ-ًؚٟ]*", '', text)
    text = re.sub(r"\s{2,}", ' ', text)
    text = re.sub(r"[\u064B-\u0652]", '', text)
    text = re.sub(r"[إأآا]", 'ا', text)
    return text.replace('ة','ه').replace('ى','ي').replace('ؤ','و').replace('ئ','ي').strip()

def clean_and_stem_arabic(text):
    tokens = re.sub(r"[^\u0600-\u06FF\s]", '', text).split()
    filtered = [t for t in tokens if t not in arabic_stopwords]
    return ' '.join([stemmer.stem(word) for word in filtered])

def encode_Sbert(questions, answers):
    questions = [clean_and_stem_arabic(clean_text(q)) for q in questions]
    answers = [clean_and_stem_arabic(clean_text(a)) for a in answers]
    q_embeddings = Sbert.encode(questions, convert_to_tensor=True, normalize_embeddings=True)
    a_embeddings = Sbert.encode(answers, convert_to_tensor=True, normalize_embeddings=True)
    similarities = cos_sim(q_embeddings, a_embeddings).diagonal().tolist()
    return pd.DataFrame([similarities], columns=[f"Q{i+1}_sim" for i in range(len(similarities))])

def get_score(model, X_test):
    return model.predict_proba(X_test)

def analyze_user_responses(questions, answers):
    encoded = encode_Sbert(questions, answers)
    dep_probs = get_score(rfc_dep, encoded)[0]
    anx_probs = get_score(rfc_anx, encoded)[0]
    return {
        "Depression": round(dep_probs[1]*100, 2),
        "Anxiety": round(anx_probs[1]*100, 2),
        "Healthy (Depression Model)": round(dep_probs[0]*100, 2),
        "Healthy (Anxiety Model)": round(anx_probs[0]*100, 2)
    }
# ----------------- Page Setup -----------------
st.set_page_config(page_title="منصة طَمّني", layout="centered")

# ----------------- Arabic Styling -----------------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap');

    html, body, .stApp {
        font-family: 'Cairo', sans-serif;
        direction: rtl;
        background: linear-gradient(90deg, #e2e2e2, #c9d6ff);
        padding: 0;
        margin: 0;
    }

    .header-box {
        background: #2a4d9f;
        border-radius: 30px;
        box-shadow: 0 0 30px rgba(0, 0, 0, 0.1);
        width: 80%;
        max-width: 850px;
        margin: 40px auto 10px;
        padding: 30px;
        color: white;
        text-align: center;
    }

    .title-inside {
        font-size: 40px;
        font-weight: 700;
        margin-bottom: 0;
    }

    .note-box {
        background: white;
        border-radius: 20px;
        padding: 20px;
        margin: 10px auto;
        width: 80%;
        max-width: 850px;
        color: #444;
        text-align: center;
        font-size: 16px;
    }

    .sub-box {
        background: white;
        border-radius: 20px;
        padding: 30px;
        margin: 10px auto 30px;
        width: 80%;
        max-width: 850px;
        color: black;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------- Static UI -----------------
st.markdown('<div class="header-box"><div class="title-inside">منصة طَمّني</div></div>', unsafe_allow_html=True)
st.markdown('<div class="note-box">هذه المنصة لا تُغني عن تشخيص الطبيب المختص، بل تهدف إلى دعم قراره بشكل مبدئي.</div>', unsafe_allow_html=True)

# ----------------- Login/Register Interface -----------------
if "page" not in st.session_state:
    st.session_state.page = "login"

if st.session_state.page == "login":
    action = st.radio("اختر الإجراء", ["تسجيل الدخول", "تسجيل جديد"], horizontal=True, key="action_selector")

    if action == "تسجيل الدخول":
        username = st.text_input("اسم المستخدم", key="login_username")
        password = st.text_input("كلمة المرور", type="password", key="login_password")

        if st.button("دخول", key="login_btn"):
            user = users_col.find_one({"username": username, "password": password})
            if user:
                st.session_state.user = username
                st.session_state.page = "questions"
                st.rerun()
            else:
                st.error("اسم المستخدم أو كلمة المرور غير صحيحة.")

    elif action == "تسجيل جديد":
        new_username = st.text_input("اسم مستخدم جديد", key="register_username")
        new_password = st.text_input("كلمة مرور جديدة", type="password", key="register_password")

        if st.button("تسجيل", key="register_btn"):
            if users_col.find_one({"username": new_username}):
                st.warning("اسم المستخدم مسجل مسبقاً.")
            else:
                users_col.insert_one({"username": new_username, "password": new_password})
                st.success("تم إنشاء الحساب بنجاح. يمكنك الآن تسجيل الدخول.")

# ----------------- Questionnaire -----------------
def questionnaire():
    st.markdown('<div class="header-box">', unsafe_allow_html=True)
    st.markdown('<div class="title-inside">التقييم النفسي</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    gender = st.radio("ما هو جنسك؟", ["ذكر", "أنثى"])
    age = st.radio("ما هي فئتك العمرية؟", ["18-29", "30-39", "40-49", "50+"])


    questions = [
        """س1: هل مررت بفترة استمرت أسبوعين أو أكثر كنت تعاني خلالها من خمسة أعراض أو أكثر مما يلي، مع ضرورة وجود عرض المزاج المكتئب أو فقدان الشغف والاهتمام بالأنشطة التي كنت تستمتع بها سابقًا؟
الأعراض تشمل: الشعور بمزاج مكتئب معظم ساعات اليوم يوميًا على مدى أسبوعين أو أكثر (مثل الحزن، فقدان الأمل، الشعور بالفراغ، أو البكاء المتكرر)، الإحساس المستمر بالتعب والإرهاق، فقدان واضح للشغف أو الاهتمام بالقيام بالواجبات أو الأنشطة اليومية، تغير في الشهية (زيادة أو نقصان) أو الوزن، صعوبة في النوم أو زيادة في عدد ساعات النوم، الشعور بالخمول الذهني أو الحركي أو على العكس، وجود نشاط حركي غير هادف ومبعثر، الشعور بفقدان القيمة الذاتية أو تأنيب ضمير مبالغ فيه، صعوبة في التركيز أو اتخاذ القرارات، وجود أفكار متكررة تتعلق بتمني الموت أو التفكير بالانتحار. اذكر الأعراض التي عانيت منها بالتفصيل وكيف أثرت عليك؟""",

        """س2: هل أدت الأعراض التي مررت بها إلى شعورك بضيق نفسي شديد أو إلى تعطيل واضح لقدرتك على أداء مهامك اليومية، سواء في حياتك الاجتماعية، الوظيفية، أو الشخصية؟ كيف لاحظت تأثير ذلك عليك وعلى تفاعلاتك مع من حولك؟""",

        """س3: هل هذه الأعراض التي عانيت منها لم تكن ناتجة عن تأثير أي مواد مخدرة، أدوية معينة، أو بسبب حالة مرضية عضوية أخرى قد تكون أثرت على سلوكك أو مشاعرك خلال تلك الفترة؟""",

        """س4: هل تجد نفسك تعاني من التفكير المفرط أو القلق الزائد تجاه مختلف الأمور الحياتية المحيطة بك، سواء كانت متعلقة بالعمل، الدراسة، المنزل، أو غيرها من الجوانب اليومية؟ أعط أمثلة على بعض من هذه الأمور وكيف يؤثر التفكير والقلق بها على أفكارك وسلوكك خلال اليوم؟""",

        """س5: هل تواجه صعوبة في السيطرة على أفكارك القلقة أو التحكم في مستوى القلق الذي تشعر به، بحيث تشعر أن الأمر خارج عن إرادتك أو أنه مستمر على نحو يرهقك؟ اجعل إجابتك تفصيلية بحيث توضح كيف يكون خارج عن إرادتك أو إلى أي مدى يرهقك.""",

        """س6: هل يترافق مع التفكير المفرط أو القلق المستمر ثلاثة أعراض أو أكثر من الأعراض التالية: الشعور بعدم الارتياح أو بضغط نفسي كبير، الإحساس بالتعب والإرهاق بسهولة، صعوبة واضحة في التركيز، الشعور بالعصبية الزائدة، شد عضلي مزمن، اضطرابات في النوم، وغيرها؟ 
اذكر كل عرض تعاني منه وهل يؤثر على مهامك اليومية مثل العمل أو الدراسة أو حياتك الاجتماعية؟ وكيف يؤثر عليك بشكل يومي؟"""
    ]
    answers = [st.text_area(q, key=f"q{i}") for i, q in enumerate(questions)]

    if st.button("إرسال"):
        if all(ans.strip() for ans in answers):
            result = analyze_user_responses(questions, answers)
            diagnosis_text = f"""
            - نسبة الاكتئاب: {result['Depression']}٪  
            - نسبة القلق: {result['Anxiety']}٪  
            - نسبة السليم (نموذج الاكتئاب): {result['Healthy (Depression Model)']}٪  
            - نسبة السليم (نموذج القلق): {result['Healthy (Anxiety Model)']}٪
            """

            responses_col.insert_one({
                "username": st.session_state.get("user", "مستخدم مجهول"),
                "gender": gender,
                "age": age,
                **{f"q{i+1}": ans for i, ans in enumerate(answers)},
                "result": diagnosis_text,
                "timestamp": datetime.now()
            })
            st.session_state.result_text = diagnosis_text
            st.session_state.page = "result"
            st.rerun()
        else:
            st.error("يرجى تعبئة جميع الإجابات.")

def show_results():
    st.markdown('<div class="header-box"><div class="title-inside">تم استلام تقييمك</div></div>', unsafe_allow_html=True)
    st.success("شكراً لمشاركتك. فيما يلي نتائج التحليل:")
    st.info(st.session_state.get("result_text", "لا توجد نتائج."))

# ----------------- Routing -----------------
if st.session_state.page == "login":
    # show_login_register()
    pass  # (insert login code)

elif st.session_state.page == "questions":
    questionnaire()

elif st.session_state.page == "result":
    show_results()
