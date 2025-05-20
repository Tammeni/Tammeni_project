import streamlit as st
st.set_page_config(page_title="منصة طَمّني", layout="centered")

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
import os




# MongoDB Connection
uri = "mongodb+srv://tammeni25:mentalhealth255@tamminicluster.nunk6nw.mongodb.net/?retryWrites=true&w=majority&authSource=admin"
client = MongoClient(uri)
db = client["tammini_db"]
users_col = db["users"]
responses_col = db["responses"]

#Donwload M @st.cache_resource ---
@st.cache_resource
def load_models():
    svm_dep = joblib.load("Depression.pkl")
    svm_anx = joblib.load("Anxiety.pkl")
    return svm_dep, svm_anx

@st.cache_resource
def load_sbert_model():
    model_path = os.path.join(os.getcwd(), 'sbert_model5')
    model = SentenceTransformer(model_path)

    # Optional: freeze model & force CPU
    for module in model._modules.values():
        for param in module.parameters():
            param.requires_grad = False
    model._target_device = torch.device("cpu")
    return model


# --- Download M one time ---
svm_dep, svm_anx = load_models()
Sbert = load_sbert_model()

# --- Clean text ---
def clean_text(text):
    cleaned = re.sub(r"[\'\"\d,;.،؛.؟]", ' ', text)
    cleaned = re.sub(r"\s{2,}", ' ', cleaned)
    emoji_pattern = re.compile(
        "[" +
        u"\U0001F600-\U0001F64F" +  # emoticons
        u"\U0001F300-\U0001F5FF" +  # symbols & pictographs
        u"\U0001F680-\U0001F6FF" +  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF" +  # flags
        u"\U00002702-\U000027B0" +  # dingbats
        u"\U000024C2-\U0001F251" +  # enclosed characters
        "]", flags=re.UNICODE
    )
    cleaned = emoji_pattern.sub(r'', cleaned)
    cleaned = re.sub(r'[\u064B-\u0652]', '', cleaned)
    cleaned = re.sub(r'[إأآا]', 'ا', cleaned)
    cleaned = cleaned.replace('ة','ه').replace('ى','ي').replace('ؤ','و').replace('ئ','ي')
    return cleaned.strip()
#----------------------------
def is_arabic_only(text):
    arabic_pattern = re.compile(r"^[\u0600-\u06FF\s\u064B-\u0652،؟؛.،.!؟]*$")
    return bool(arabic_pattern.fullmatch(text.strip()))

# ---Encrypt text by SBERT ---
def encode_Sbert(questions, answers):
    questions = [clean_text(q) for q in questions]
    answers = [clean_text(a) for a in answers]
    q_emb = Sbert.encode(questions, convert_to_tensor=True, normalize_embeddings=True)
    a_emb = Sbert.encode(answers, convert_to_tensor=True, normalize_embeddings=True)
    similarities = cos_sim(q_emb, a_emb).diagonal().tolist()
    return pd.DataFrame([similarities], columns=[f"Q{i+1}_sim" for i in range(len(similarities))])

# ---  analysis ---
def get_score(model, X_test):
    return model.predict_proba(X_test)

# --- analysis Answr ---
def analyze_user_responses(answers, questions):
    questions_dep = questions[:3]
    questions_anx = questions[2:6]
    dep_encoded = encode_Sbert(questions_dep, answers[:3])
    dep_score = get_score(svm_dep, dep_encoded)[0]
    anx_encoded = encode_Sbert(questions_anx, answers[2:6])
    anx_score = get_score(svm_anx, anx_encoded)[0]
    return {
        "Depression": int(dep_score[0] * 100),
        "Anxiety": int(anx_score[0] * 100)
    }

# --- gui Desi ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo :wght@400;700&display=swap');
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

# --- main menu ---
st.markdown('<div class="header-box"><div class="title-inside">منصة طَمّني</div></div>', unsafe_allow_html=True)
st.markdown('<div class="note-box">هذه المنصة لا تُغني عن تشخيص الطبيب المختص، بل تهدف إلى دعم قراره بشكل مبدئي.</div>', unsafe_allow_html=True)

# ---login ---
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

# --- Question ---
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
    answers = []
for i, q in enumerate(questions):
    answers.append(st.text_area(f"{q}", key=f"q{i}"))

if st.button("إرسال"):
    if not all(ans.strip() for ans in answers):
        st.error(" يرجى تعبئة جميع الإجابات.")
    elif not all(is_arabic_only(ans) for ans in answers):
        st.error("يُسمح فقط باستخدام الحروف العربية في الإجابات.")
    else:   
        responses_col.insert_one({
            "username": st.session_state.get("user", "مستخدم مجهول"),
            "gender": gender,
            "age": age,
            **{f"q{i+1}": ans for i, ans in enumerate(answers)},
            "result": "قيد المعالجة",
            "timestamp": datetime.now()
        })
        result = analyze_user_responses(answers, questions)
        latest_doc = responses_col.find_one(
            {"username": st.session_state.get("user")},
            sort=[("timestamp", -1)]
        )
        if latest_doc:
            responses_col.update_one(
                {"_id": latest_doc["_id"]},
                {"$set": {
                    "نسبة الاكتئاب": result["Depression"],
                    "نسبة القلق": result["Anxiety"],
                    "result": "تم التحليل"
                }}
            )
        st.session_state.page = "result"
        st.rerun()


if st.session_state.page == "questions":
    if st.button(" عرض الإجابات السابقة"):
        st.session_state.page = "history"
        st.rerun()

    questionnaire()
    elif st.session_state.page == "result":
    elif st.session_state.page == "history":
    st.markdown('<div class="header-box"><div class="title-inside">الإجابات السابقة</div></div>', unsafe_allow_html=True)

    user_past = list(responses_col.find(
        {"username": st.session_state.get("user")},
        sort=[("timestamp", -1)]
    ))

    if not user_past:
        st.info("لا توجد نتائج سابقة محفوظة لهذا المستخدم.")
    else:
        for i, entry in enumerate(user_past[:5]):
            st.markdown(f"---\n#### المحاولة رقم {i+1}")
            st.markdown(f"**التاريخ**: `{entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}`")
            st.markdown(f"**الجنس**: {entry.get('gender', 'غير محدد')}  |  **العمر**: {entry.get('age', 'غير محدد')}")
            st.markdown("**الأجوبة:**")
            for j in range(1, 7):
                q_text = f"q{j}"
                if q_text in entry:
                    st.markdown(f"- **س{j}**: {entry[q_text]}")
            st.markdown(f"🔹 **نسبة الاكتئاب**: `{entry.get('نسبة الاكتئاب', 'N/A')}%`")
            st.markdown(f"🔹 **نسبة القلق**: `{entry.get('نسبة القلق', 'N/A')}%`")
            st.markdown(f"📌 **الحالة**: `{entry.get('result', 'قيد المعالجة')}`")

    if st.button("🔙 العودة إلى التقييم"):
        st.session_state.page = "questions"
        st.rerun()

    latest_doc = responses_col.find_one(
        {"username": st.session_state.get("user")},
        sort=[("timestamp", -1)]
    )
    if latest_doc:
        answers = [
            latest_doc.get("q1", ""),
            latest_doc.get("q2", ""),
            latest_doc.get("q3", ""),
            latest_doc.get("q4", ""),
            latest_doc.get("q5", ""),
            latest_doc.get("q6", "")
        ]
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
        result = analyze_user_responses(answers, questions)
        responses_col.update_one(
            {"_id": latest_doc["_id"]},
            {"$set": {
                "نسبة الاكتئاب": result["Depression"],
                "نسبة القلق": result["Anxiety"],
                "result": "تم التحليل"
            }}
        )
        st.markdown('<div class="header-box">', unsafe_allow_html=True)
        st.markdown('<div class="title-inside">نتيجة التحليل</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.success("✅ تم تحليل إجاباتك بنجاح بواسطة نموذج الذكاء الاصطناعي.")
        st.markdown(f"""
        ### 🧠 نتائج التحليل:
        -  **نسبة الاكتئاب**: {result['Depression']}%
        -  **نسبة القلق**: {result['Anxiety']}%
        📌 **تنويه**: هذه النسب تقديرية فقط، ويُفضل استشارة مختص نفسي لتأكيد التشخيص.
        """)
    else:
        st.warning("لم يتم العثور على إجابات لعرضها.")
