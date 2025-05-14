# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import streamlit as st
from pymongo import MongoClient
from datetime import datetime
import pandas as pd
import joblib
import os
from sklearn.preprocessing import LabelEncoder

# Import functions from pipeline.py
from pipeline_project import encode_Sbert, clean_text, clean_and_stem_arabic, get_score, translate_to_fusha, wrap_answers_in_df

# ----------------- Load Trained Models -----------------
model_path = os.getcwd()
rfc_dep = joblib.load(os.path.join(model_path, 'rfc_dep (1).pkl'))
rfc_anx = joblib.load(os.path.join(model_path, 'rfc_anx (1).pkl'))

DepEncoder = LabelEncoder()
DepEncoder.classes_ = ["Depression", "Healthy"]

AnxEncoder = LabelEncoder()
AnxEncoder.classes_ = ["Anxiety", "Healthy"]

# ----------------- Database Connection -----------------
uri = "mongodb+srv://tammeni25:mentalhealth255@tamminicluster.nunk6nw.mongodb.net/?retryWrites=true&w=majority&authSource=admin"
client = MongoClient(uri)
db = client["tammini_db"]
users_col = db["users"]
responses_col = db["responses"]

# ----------------- Page Config -----------------
st.set_page_config(page_title="منصة طَمّني", layout="centered")

# ----------------- Landing Page -----------------
def show_landing_page():
    st.markdown("""
        <style>
        html, body, .stApp {
            background-color: #e6f7ff;
            font-family: 'Cairo', sans-serif;
            direction: rtl;
        }
        .landing-container {
            text-align: center;
            padding: 40px;
            background-color: #d4ecf7;
            border-radius: 16px;
            margin-bottom: 30px;
            box-shadow: 0px 0px 15px rgba(0, 91, 153, 0.1);
        }
        h1 {
            color: #005b99;
            font-size: 48px;
            margin-bottom: 10px;
        }
        h3 {
            color: #333;
            font-size: 20px;
            margin-bottom: 10px;
        }
        .note {
            font-size: 18px;
            color: #005b99;
            background-color: #e6f2ff;
            padding: 10px;
            border-radius: 8px;
            display: inline-block;
            margin-top: 15px;
        }
        </style>
        <div class='landing-container'>
            <h1>طَمّني</h1>
            <h3>منصة تقييم الصحة النفسية باستخدام الذكاء الاصطناعي</h3>
            <div class='note'>هذه المنصة لا تُعد بديلاً عن الطبيب، بل تُساعد الأطباء في اتخاذ قراراتهم</div>
        </div>
    """, unsafe_allow_html=True)

    if st.button("تسجيل الدخول / إنشاء حساب"):
        st.session_state.page = "auth"

# ----------------- Auth -----------------
def signup():
    st.title("تسجيل حساب جديد")
    username = st.text_input("اسم المستخدم")
    password = st.text_input("كلمة المرور", type="password")
    if st.button("تسجيل"):
        if users_col.find_one({"username": username}):
            st.warning("هذا المستخدم مسجل بالفعل. يتم عرض التقرير الأول:")
            record = responses_col.find_one({"username": username}, sort=[("timestamp", 1)])
            if record:
                for i in range(1, 7):
                    st.write(f"س{i}: {record.get(f'q{i}', '')}")
                st.success(f"النتيجة: {record.get('result', 'لم يتم التحليل')}")
            else:
                st.info("لا توجد ردود محفوظة.")
        else:
            users_col.insert_one({"username": username, "password": password})
            st.success("تم التسجيل بنجاح. يمكنك تسجيل الدخول الآن.")

def login():
    st.title("تسجيل الدخول")
    username = st.text_input("اسم المستخدم")
    password = st.text_input("كلمة المرور", type="password")
    if st.button("دخول"):
        user = users_col.find_one({"username": username, "password": password})
        if user:
            st.session_state.user = username
            st.success("تم تسجيل الدخول.")
            st.session_state.page = "questionnaire"
        else:
            st.error("اسم المستخدم أو كلمة المرور غير صحيحة.")

# ----------------- Questionnaire -----------------
def questionnaire():
    st.subheader("التقييم النفسي")
    gender = st.radio("ما هو جنسك؟", ["ذكر", "أنثى"])
    age = st.radio("ما هي فئتك العمرية؟", ["18-29", "30-39", "40-49", "50+"]) 
    questions = {
        1: """س1: هل مررت بفترة استمرت أسبوعين أو أكثر كنت تعاني خلالها من خمسة أعراض أو أكثر مما يلي، مع ضرورة وجود عرض المزاج المكتئب أو فقدان الشغف والاهتمام بالأنشطة التي كنت تستمتع بها سابقًا؟
              5 الأعراض تشمل: الشعور بمزاج مكتئب معظم ساعات اليوم يوميًا على مدى أسبوعين أو أكثر  (مثل الحزن، فقدان الأمل، الشعور بالفراغ، أو البكاء المتكرر)،  الإحساس المستمر بالتعب والإرهاق، فقدان واضح للشغف أو الاهتمام بالقيام بالواجبات أو الأنشطة اليومية، تغير في الشهية (زيادة أو نقصان) أو الوزن بأكثر من %، صعوبة في النوم أو زيادة في عدد ساعات النوم،  الشعور بالخمول الذهني أو الحركي أو على العكس، وجود نشاط حركي غير هادف ومبعثر، الشعور بفقدان القيمة الذاتية أو تأنيب ضمير مبالغ فيه،  صعوبة في التركيز أو اتخاذ القرارات،  وجود أفكار متكررة تتعلق بتمني الموت أو التفكير بالانتحار. اذكر الأعراض التي عانيت منها بالتفصيل وكيف أثرت عليك؟""",

        2: """س2: هل أدت الأعراض التي مررت بها إلى شعورك بضيق نفسي شديد أو إلى تعطيل واضح لقدرتك على أداء مهامك اليومية، 
               سواء في حياتك الاجتماعية، الوظيفية، أو الشخصية؟ كيف لاحظت تأثير ذلك عليك وعلى تفاعلاتك مع من حولك؟""",

        3: """س3: هل هذه الأعراض التي عانيت منها لم تكن ناتجة عن تأثير أي مواد مخدرة، 
              أدوية معينة، أو بسبب حالة مرضية عضوية أخرى قد تكون أثرت على سلوكك أو مشاعرك خلال تلك الفترة؟""",

        4: """س4: هل تجد نفسك تعاني من التفكير المفرط أو القلق الزائد تجاه مختلف الأمور الحياتية المحيطة بك،  سواء كانت متعلقة بالعمل، الدراسة، المنزل، أو غيرها من الجوانب اليومية؟ 
                   اعط أمثلة على بعض من هذه الأمور وكيف يؤثر التفكير والقلق بها على أفكارك وسلوكك خلال اليوم ؟""",

        5: """س5: هل تواجه صعوبة في السيطرة على أفكارك القلقة أو التحكم في مستوى القلق الذي تشعر به، 
                    بحيث تشعر أن الأمر خارج عن إرادتك أو أنه مستمر على نحو يرهقك؟ اجعل إجابتك تفصيلية بحيث توضح كيف يكون خارج عن إرادتك أو إلى أي مدى يرهقك.""",

        6: """س6: هل يترافق مع التفكير المفرط أو القلق المستمر ثلاثة أعراض أو أكثر من الأعراض التالية:  الشعور بعدم الارتياح أو بضغط نفسي كبير، الإحساس بالتعب والإرهاق بسهولة،  صعوبة واضحة في التركيز، الشعور بالعصبية الزائدة، شد عضلي مزمن، اضطرابات في النوم، وغيرها؟ 
               اذكر كل عرض تعاني منه وهل يؤثر على مهامك اليومية مثل العمل أو الدراسة أو حياتك الاجتماعية؟  وكيف يؤثر عليك بشكل يومي؟"""
    }
    answers = {}
    for i in range(1, 7):
        answers[f"q{i}"] = st.text_area(f"س{i}: {questions[i]}")

    if st.button("إرسال التقييم"):
        if any(a.strip() == "" for a in answers.values()):
            st.error("يرجى الإجابة على جميع الأسئلة.")
            return
        user = st.session_state.get("user")
        if not user:
            st.error("يرجى تسجيل الدخول أولًا.")
            return

        try:
            # Prepare answers
            responses = [translate_to_fusha(answers[f"q{i}"]) for i in range(1, 7)]
            questions_list = [f"س{i}" for i in range(1, 7)]
            answers_df = wrap_answers_in_df(questions_list, responses)
            encoded = encode_Sbert(questions_list, answers_df)

            # Predictions
            dep_pred = rfc_dep.predict(encoded)[0]
            anx_pred = rfc_anx.predict(encoded)[0]
            dep_label = DepEncoder.inverse_transform([dep_pred])[0]
            anx_label = AnxEncoder.inverse_transform([anx_pred])[0]

            # Final diagnosis
            if dep_label == "Depression" and anx_label == "Anxiety":
                final_result = "كلا الاكتئاب والقلق"
            elif dep_label == "Depression":
                final_result = "اكتئاب"
            elif anx_label == "Anxiety":
                final_result = "قلق"
            else:
                final_result = "سليم / لا توجد مؤشرات واضحة"

            # Store in DB
            responses_col.insert_one({
                "username": user,
                "gender": gender,
                "age": age,
                **answers,
                "result": final_result,
                "timestamp": datetime.now()
            })

            # Show result
            st.success(f"✅ تم تحليل النتيجة: {final_result}")
            st.session_state.latest_result = final_result
            st.session_state.page = "result"
            st.experimental_rerun()

        except Exception as e:
            st.error(f"خطأ أثناء التحليل: {str(e)}")

# ----------------- Result Page -----------------
def show_results():
    st.title("نتيجة التقييم")
    if "latest_result" in st.session_state:
        st.success(st.session_state.latest_result)
    else:
        st.info("لا توجد نتائج متاحة حالياً.")
   

   
#  ----------------- Navigation -----------------
if "page" not in st.session_state:
    st.session_state.page = "landing"

if st.session_state.page == "landing":
    show_landing_page()
elif st.session_state.page == "auth":
    choice = st.radio("اختر", ["تسجيل الدخول", "تسجيل جديد"], horizontal=True)
    if choice == "تسجيل الدخول":
        login()
    else:
        signup()
elif st.session_state.page == "questionnaire":
    questionnaire()
elif st.session_state.page == "result":
    show_results()
