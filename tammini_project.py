import streamlit as st
from pymongo import MongoClient
from datetime import datetime

# ----------------- MongoDB Connection -----------------
uri = "mongodb+srv://tammeni25:mentalhealth255@tamminicluster.nunk6nw.mongodb.net/?retryWrites=true&w=majority&authSource=admin"
client = MongoClient(uri)
db = client["tammini_db"]
users_col = db["users"]
responses_col = db["responses"]

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

    .container-box {
        background: #2a4d9f;
        border-radius: 30px;
        box-shadow: 0 0 30px rgba(0, 0, 0, 0.1);
        width: 80%;
        max-width: 850px;
        margin: 40px auto;
        padding: 40px;
        color: white;
        text-align: center;
    }

    .title {
        font-size: 32px;
        font-weight: 700;
        color: white;
        margin-bottom: 10px;
    }

    .sub-box {
        background: white;
        border-radius: 20px;
        padding: 30px;
        margin-top: 30px;
        color: black;
    }

    .note {
        margin-top: 10px;
        font-size: 14px;
        color: #eee;
    }

    </style>
""", unsafe_allow_html=True)

# ----------------- State Setup -----------------
if "page" not in st.session_state:
    st.session_state.page = "login"

# ----------------- Login/Register Interface -----------------
if st.session_state.page == "login":
    st.markdown('''<div class="container-box">''', unsafe_allow_html=True)
    st.markdown('<div class="title">منصة طَمّني</div>', unsafe_allow_html=True)
    st.markdown('<div class="note">هذه المنصة لا تُغني عن تشخيص الطبيب المختص، بل تهدف إلى دعم قراره بشكل مبدئي.</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-box">', unsafe_allow_html=True)

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

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ----------------- Questionnaire -----------------
def questionnaire():
    st.markdown('<div class="container-box">', unsafe_allow_html=True)
    st.markdown('<div class="title">التقييم النفسي</div>', unsafe_allow_html=True)

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
        if all(ans.strip() for ans in answers):
            responses_col.insert_one({
                "username": st.session_state.get("user", "مستخدم مجهول"),
                "gender": gender,
                "age": age,
                **{f"q{i+1}": ans for i, ans in enumerate(answers)},
                "result": "قيد المعالجة",
                "timestamp": datetime.now()
            })
            st.session_state.page = "result"
            st.rerun()
        else:
            st.error("يرجى تعبئة جميع الإجابات.")
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------- Main Page Routing -----------------

if st.session_state.page == "questions":
    questionnaire()

elif st.session_state.page == "result":
    st.markdown('<div class="container-box">', unsafe_allow_html=True)
    st.markdown('<div class="title">تم استلام تقييمك</div>', unsafe_allow_html=True)
    st.success("شكراً لمشاركتك. سيتم عرض النتيجة بعد تحليل البيانات.")
    st.markdown('</div>', unsafe_allow_html=True)
