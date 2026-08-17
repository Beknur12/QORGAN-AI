import streamlit as st
import time
import re
import joblib
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter

# ============================================================
# QORGAN AI
# ============================================================

st.set_page_config(
    page_title="QORGAN AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)
# ============================================================
# ML MODEL
# ============================================================

@st.cache_resource
def load_ml_model():
    model = joblib.load("models/qorgan_model.pkl")
    vectorizer = joblib.load("models/qorgan_vectorizer.pkl")
    return model, vectorizer


model, vectorizer = load_ml_model()

import os
import shutil


# ============================================================
# TESSERACT — WINDOWS + CLOUD
# ============================================================

def configure_tesseract():
    # Windows-тағы локалды Tesseract
    windows_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    if os.path.exists(windows_path):
        pytesseract.pytesseract.tesseract_cmd = windows_path
        return

    # Linux / Streamlit Community Cloud
    cloud_path = shutil.which("tesseract")

    if cloud_path:
        pytesseract.pytesseract.tesseract_cmd = cloud_path
        return

    raise RuntimeError(
        "Tesseract OCR жүйеден табылмады."
    )


configure_tesseract()

def extract_text_from_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    width, height = image.size
    image = image.resize((width * 2, height * 2)).convert("L")
    image = ImageEnhance.Contrast(image).enhance(1.7)
    image = image.filter(ImageFilter.SHARPEN)
    text = pytesseract.image_to_string(
        image, lang="kaz+rus+eng",
        config="--oem 3 --psm 6 -c preserve_interword_spaces=1"
    )
    return " ".join(text.split())

def analyze_message(message):
    vector = vectorizer.transform([message])
    prediction = int(model.predict(vector)[0])
    probabilities = model.predict_proba(vector)[0]
    idx = {int(label): i for i, label in enumerate(model.classes_)}
    score = round(float(probabilities[idx[1]]) * 100)
    text = message.lower()
    indicators = []
    if re.search(r"https?://|www\.|bit\.ly|tinyurl|t\.me/", text):
        indicators.append("Хабарламада сыртқы сілтеме анықталды")
    if any(w in text for w in ["дереу","қазір","срочно","немедленно","соңғы мүмкіндік","последний шанс"]):
        indicators.append("Асықтыру немесе психологиялық қысым белгісі бар")
    if any(w in text for w in ["cvv","пароль","құпиясөз","sms код","смс код"]):
        indicators.append("Құпия немесе жеке деректерге қатысты сөздер бар")
    if any(w in text for w in ["банк","bank","карта","card","ақша","деньги"]):
        indicators.append("Қаржылық ақпаратқа қатысты сөздер анықталды")
    return prediction, score, indicators

def show_analysis(message):
    prediction, score, indicators = analyze_message(message)
    st.divider()
    st.subheader("Талдау нәтижесі")
    left, right = st.columns(2)
    with left:
        st.metric("Алаяқтық ықтималдығы", f"{score}%")
    with right:
        st.metric("ML шешімі", "Алаяқтық" if prediction == 1 else "Қауіпсіз")
    if score >= 70:
        st.error("Жоғары қауіп деңгейі — бұл хабарламада алаяқтық белгілері жоғары.")
    elif score >= 40:
        st.warning("Орташа қауіп деңгейі — хабарламаны қосымша тексерген жөн.")
    else:
        st.success("Төмен қауіп деңгейі — айқын алаяқтық белгілері аз.")
    if indicators:
        st.markdown("#### Анықталған белгілер")
        for item in indicators:
            st.write(f"• {item}")
    else:
        st.caption("Қосымша ереже-негізіндегі қауіп белгілері анықталған жоқ.")
    st.caption("QORGAN AI • TF-IDF + Naive Bayes")

# ============================================================
# CSS
# ============================================================

st.markdown(
    """
<style>

/* ---------- COLORS ---------- */

:root {
    --bg-main: #020b16;
    --bg-second: #031525;
    --cyan: #00e5ff;
    --cyan-soft: rgba(0, 229, 255, 0.20);
    --green: #00e89d;
    --yellow: #ffc94a;
    --white: #eefcff;
    --muted: #7896aa;
}


/* ---------- GLOBAL ---------- */

html,
body,
[class*="css"] {
    font-family: "Segoe UI", Arial, sans-serif;
}

.stApp {
    color: var(--white);

    background:
        radial-gradient(
            circle at 12% 18%,
            rgba(0, 229, 255, 0.10),
            transparent 25%
        ),
        radial-gradient(
            circle at 88% 15%,
            rgba(0, 105, 255, 0.08),
            transparent 28%
        ),
        radial-gradient(
            circle at 50% 100%,
            rgba(0, 229, 255, 0.04),
            transparent 30%
        ),
        linear-gradient(
            135deg,
            #010711 0%,
            #031422 48%,
            #020914 100%
        );

    min-height: 100vh;
}


/* ---------- MOVING GRID ---------- */

.stApp::before {
    content: "";

    position: fixed;
    inset: 0;

    pointer-events: none;

    background-image:
        linear-gradient(
            rgba(0, 229, 255, 0.023) 1px,
            transparent 1px
        ),
        linear-gradient(
            90deg,
            rgba(0, 229, 255, 0.023) 1px,
            transparent 1px
        );

    background-size: 52px 52px;

    animation: gridMove 24s linear infinite;

    z-index: 0;
}

@keyframes gridMove {

    from {
        background-position: 0 0;
    }

    to {
        background-position: 52px 52px;
    }
}


/* ---------- STREAMLIT ---------- */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    background: transparent !important;
}

.block-container {
    position: relative;
    z-index: 2;

    max-width: 1260px;

    padding-top: 2rem;
    padding-bottom: 2rem;
}


/* ---------- TEXT ---------- */

h1,
h2,
h3 {
    color: #f3fcff !important;
}

h1 {
    font-weight: 850 !important;
    letter-spacing: -1px !important;
}

h2 {
    font-weight: 800 !important;
}

h3 {
    font-weight: 750 !important;
}

p {
    color: #8ca7b8;
}

[data-testid="stCaptionContainer"] {
    color: #68899e;
}


/* ==========================================================
   NATIVE STREAMLIT CARDS
========================================================== */

[data-testid="stVerticalBlockBorderWrapper"] {
    position: relative;

    overflow: hidden;

    border:
        1px solid rgba(0, 229, 255, 0.17) !important;

    border-radius: 18px !important;

    background:
        linear-gradient(
            145deg,
            rgba(5, 25, 44, 0.82),
            rgba(2, 14, 27, 0.90)
        ) !important;

    box-shadow:
        0 18px 50px rgba(0, 0, 0, 0.20);

    backdrop-filter: blur(14px);

    transition:
        transform 0.28s ease,
        border-color 0.28s ease,
        box-shadow 0.28s ease;
}

[data-testid="stVerticalBlockBorderWrapper"]:hover {
    transform: translateY(-3px);

    border-color:
        rgba(0, 229, 255, 0.40) !important;

    box-shadow:
        0 20px 50px rgba(0, 0, 0, 0.25),
        0 0 28px rgba(0, 229, 255, 0.055);
}


/* ---------- GLOWING DOT ON CARDS ---------- */

[data-testid="stVerticalBlockBorderWrapper"]::before {
    content: "";

    position: absolute;

    width: 7px;
    height: 7px;

    top: 15px;
    right: 16px;

    border-radius: 50%;

    background: #00e5ff;

    box-shadow:
        0 0 5px #00e5ff,
        0 0 12px rgba(0, 229, 255, 0.90),
        0 0 22px rgba(0, 229, 255, 0.45);

    animation: cardDot 1.8s ease-in-out infinite;

    z-index: 5;
}

@keyframes cardDot {

    0%,
    100% {
        opacity: 1;
        transform: scale(1);
    }

    50% {
        opacity: 0.35;
        transform: scale(0.65);
    }
}


/* ==========================================================
   TEXT AREA
========================================================== */

.stTextArea label p {
    color: #aac8d9 !important;
    font-weight: 650 !important;
}

.stTextArea textarea {
    min-height: 220px !important;

    padding: 18px !important;

    border-radius: 14px !important;

    border:
        1px solid rgba(0, 229, 255, 0.25) !important;

    background:
        rgba(1, 10, 21, 0.80) !important;

    color: #effdff !important;

    font-size: 16px !important;

    line-height: 1.55 !important;

    caret-color: #00e5ff;

    transition: all 0.25s ease;
}

.stTextArea textarea:focus {
    border-color:
        rgba(0, 229, 255, 0.80) !important;

    box-shadow:
        0 0 0 1px rgba(0, 229, 255, 0.55),
        0 0 24px rgba(0, 229, 255, 0.10) !important;
}

.stTextArea textarea::placeholder {
    color: #4f7084 !important;
}


/* ==========================================================
   TABS
========================================================== */

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}

.stTabs [data-baseweb="tab"] {
    height: 46px;

    padding-left: 18px;
    padding-right: 18px;

    border-radius: 10px;

    background:
        rgba(4, 22, 39, 0.65);

    color: #7897aa;
}

.stTabs [aria-selected="true"] {
    color: #00e5ff !important;

    background:
        rgba(0, 229, 255, 0.075) !important;
}


/* ==========================================================
   FILE UPLOADER
========================================================== */

[data-testid="stFileUploader"] {
    padding: 8px;

    border-radius: 14px;

    border:
        1px solid rgba(0, 229, 255, 0.11);

    background:
        rgba(2, 14, 27, 0.50);
}


/* ==========================================================
   BUTTON
========================================================== */

.stButton > button {
    width: 100%;

    min-height: 58px;

    border-radius: 13px;

    border:
        1px solid rgba(0, 229, 255, 0.85);

    background:
        linear-gradient(
            90deg,
            rgba(0, 112, 255, 0.20),
            rgba(0, 229, 255, 0.16)
        );

    color: white;

    font-size: 16px;
    font-weight: 800;

    letter-spacing: 0.4px;

    box-shadow:
        0 0 15px rgba(0, 229, 255, 0.10);

    transition: all 0.25s ease;
}

.stButton > button:hover {
    color: white;

    border-color: #6ff7ff;

    transform: translateY(-2px);

    box-shadow:
        0 0 16px rgba(0, 229, 255, 0.35),
        0 0 35px rgba(0, 160, 255, 0.12);
}

.stButton > button:active {
    transform: scale(0.99);
}


/* ==========================================================
   METRICS
========================================================== */

[data-testid="stMetric"] {
    padding: 15px;

    border-radius: 13px;

    border:
        1px solid rgba(0, 229, 255, 0.12);

    background:
        rgba(0, 229, 255, 0.035);
}

[data-testid="stMetricValue"] {
    color: #00e5ff;
}


/* ==========================================================
   ALERTS
========================================================== */

[data-testid="stAlert"] {
    border-radius: 13px !important;
}


/* ==========================================================
   DIVIDER
========================================================== */

hr {
    border-color:
        rgba(0, 229, 255, 0.09) !important;
}


/* ==========================================================
   FOOTER
========================================================== */

.qorgan-footer {
    margin-top: 25px !important;
}


/* ==========================================================
   MOBILE
========================================================== */

@media (max-width: 768px) {

    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
        padding-top: 1rem;
    }

    h1 {
        font-size: 34px !important;
    }

    h2 {
        font-size: 24px !important;
    }

    [data-testid="stVerticalBlockBorderWrapper"]::before {
        top: 12px;
        right: 13px;
    }

}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

brand_col, status_col = st.columns(
    [2.2, 1],
    vertical_alignment="center"
)


with brand_col:

    icon_col, name_col = st.columns(
        [0.25, 2],
        vertical_alignment="center"
    )

    with icon_col:
        st.title("◈")

    with name_col:
        st.title("QORGAN AI")
        st.caption(
            "CYBER SHIELD  /  INTELLIGENT PROTECTION"
        )


with status_col:

    with st.container(border=True):

        status_dot, status_text = st.columns(
            [0.15, 1.8],
            vertical_alignment="center"
        )

        with status_dot:
            st.markdown(
                "<span style='color:#00e89d; "
                "font-size:22px;'>●</span>",
                unsafe_allow_html=True
            )

        with status_text:
            st.markdown(
                "**Қорғау жүйесі белсенді**"
            )

            st.caption(
                "SYSTEM ONLINE"
            )


# ============================================================
# HERO
# ============================================================

st.write("")
st.write("")

st.caption(
    "◈ ЖАСАНДЫ ИНТЕЛЛЕКТ  /  КИБЕРҚАУІПСІЗДІК"
)

st.title(
    "Күмәнді хабарламаны бірнеше секундта тексеріңіз"
)

st.markdown(
    """
**QORGAN AI** — қазақ және аралас тілдегі
интернет-алаяқтық хабарламаларын анықтауға арналған
интеллектуалды веб-платформа.
"""
)

st.write("")
st.write("")


# ============================================================
# MAIN AREA
# ============================================================

scanner_col, info_col = st.columns(
    [1.75, 0.85],
    gap="large"
)


# ============================================================
# SCANNER
# ============================================================

with scanner_col:

    with st.container(border=True):

        st.caption(
            "◈ QORGAN SCANNER"
        )

        st.header(
            "Күмәнді хабарламаны тексеру"
        )

        st.caption(
            "SMS, WhatsApp, Telegram немесе басқа "
            "сервистен келген күмәнді хабарламаны енгізіңіз."
        )


        text_tab, screenshot_tab = st.tabs(["Мәтін енгізу", "Скриншот жүктеу"])

        with text_tab:
            message = st.text_area(
                "Хабарлама мәтіні",
                placeholder="Мысалы: Сіздің картаңыз бұғатталды. Деректеріңізді растау үшін сілтемеге өтіңіз...",
                height=230, max_chars=3000, key="manual_message"
            )
            st.caption(f"{len(message)} / 3000 таңба")
            if st.button("QORGAN AI АРҚЫЛЫ ТЕКСЕРУ", use_container_width=True, type="primary", key="text_scan"):
                if not message.strip():
                    st.error("Алдымен тексерілетін хабарлама мәтінін енгізіңіз.")
                else:
                    with st.spinner("QORGAN AI хабарламаны талдап жатыр..."):
                        time.sleep(0.5)
                    show_analysis(message.strip())

        with screenshot_tab:
            uploaded_image = st.file_uploader(
                "Скриншотты таңдаңыз",
                type=["png", "jpg", "jpeg"],
                key="screenshot_upload"
            )
            if uploaded_image is not None:
                st.image(uploaded_image, caption="Жүктелген скриншот", use_container_width=True)
                if st.button("СКРИНШОТТЫ OCR АРҚЫЛЫ ТЕКСЕРУ", use_container_width=True, key="ocr_scan"):
                    try:
                        with st.spinner("Суреттен мәтін оқылып жатыр..."):
                            ocr_text = extract_text_from_image(uploaded_image)
                        if not ocr_text:
                            st.error("Скриншоттан мәтін табылмады. Анығырақ сурет жүктеп көріңіз.")
                        else:
                            st.markdown("#### OCR арқылы танылған мәтін")
                            st.caption("Қажет болса, танылған мәтіндегі OCR қатесін қолмен түзете аласыз.")
                            edited_ocr_text = st.text_area(
                                "Танылған мәтін",
                                value=ocr_text,
                                height=140,
                                key="ocr_result_text"
                            )
                            if edited_ocr_text.strip():
                                show_analysis(edited_ocr_text.strip())
                            else:
                                st.error("OCR нәтижесінде талдауға жарамды мәтін табылмады.")
                    except Exception as error:
                        st.error(f"OCR қатесі: {error}")


# ============================================================
# SYSTEM INFORMATION
# ============================================================

with info_col:

    with st.container(border=True):

        st.caption(
            "◈ ЖҮЙЕ КҮЙІ"
        )

        st.header(
            "QORGAN Shield"
        )

        st.markdown(
            "<span style='color:#00e89d;'>●</span> "
            "**Веб-интерфейс**",
            unsafe_allow_html=True
        )

        st.markdown(
            "<span style='color:#00e89d;'>●</span> "
            "**Мәтінді талдау**",
            unsafe_allow_html=True
        )

        st.markdown(
            "<span style='color:#00e89d;'>●</span> "
            "**Скриншот жүктеу**",
            unsafe_allow_html=True
        )

        st.markdown(
            "<span style='color:#00e89d;'>●</span> "
            "**ML классификатор**",
            unsafe_allow_html=True
        )

        st.markdown(
            "<span style='color:#00e89d;'>●</span> "
            "**OCR модулі**",
            unsafe_allow_html=True
        )


    st.write("")


    with st.container(border=True):

        st.caption(
            "◈ ҚАУІПСІЗДІК"
        )

        st.subheader(
            "Қауіпсіздік кеңесі"
        )

        st.write(
            "Күмәнді сілтемелерге өтпеңіз."
        )

        st.write(
            "Банк картасының CVV кодын, "
            "SMS-кодтарды және құпиясөздерді "
            "бөгде адамдарға бермеңіз."
        )


# ============================================================
# ANALYSIS FEATURES
# ============================================================

st.write("")
st.write("")
st.write("")

st.caption(
    "◈ QORGAN AI ТАЛДАУ ЖҮЙЕСІ"
)

st.header(
    "Платформа нені талдайды?"
)

st.caption(
    "QORGAN AI хабарламаның бірнеше белгісін "
    "бір уақытта талдауға арналған."
)

st.write("")


card1, card2, card3, card4 = st.columns(
    4,
    gap="medium"
)


# ------------------------------------------------------------
# CARD 1
# ------------------------------------------------------------

with card1:

    with st.container(border=True):

        st.caption(
            "01  /  LINK ANALYSIS"
        )

        st.subheader(
            "Күмәнді сілтемелер"
        )

        st.write(
            "URL құрылымы мен хабарламадағы "
            "сыртқы сілтемелерді талдау."
        )

        st.write("")

        st.caption(
            "СІЛТЕМЕЛЕРДІ ТЕКСЕРУ"
        )


# ------------------------------------------------------------
# CARD 2
# ------------------------------------------------------------

with card2:

    with st.container(border=True):

        st.caption(
            "02  /  URGENCY"
        )

        st.subheader(
            "Психологиялық қысым"
        )

        st.write(
            "Асықтыру, қорқыту және шұғыл "
            "әрекетке шақыру белгілерін анықтау."
        )

        st.write("")

        st.caption(
            "МӘТІНДІК БЕЛГІЛЕР"
        )


# ------------------------------------------------------------
# CARD 3
# ------------------------------------------------------------

with card3:

    with st.container(border=True):

        st.caption(
            "03  /  PRIVACY"
        )

        st.subheader(
            "Жеке деректер"
        )

        st.write(
            "Пароль, SMS-код, банк картасы "
            "және құпия ақпаратты сұрауды анықтау."
        )

        st.write("")

        st.caption(
            "ДЕРЕКТЕРДІ ҚОРҒАУ"
        )


# ------------------------------------------------------------
# CARD 4
# ------------------------------------------------------------

with card4:

    with st.container(border=True):

        st.caption(
            "04  /  AI ANALYSIS"
        )

        st.subheader(
            "Machine Learning"
        )

        st.write(
            "Қазақ және аралас тілдегі мәтінді "
            "машиналық оқыту арқылы жіктеу."
        )

        st.write("")

        st.caption(
            "ИНТЕЛЛЕКТУАЛДЫ ТАЛДАУ"
        )


# ============================================================
# FOOTER
# ============================================================

st.write("")
st.write("")
st.write("")

st.divider()

footer_left, footer_center, footer_right = st.columns(
    [1.2, 1, 1.2]
)


with footer_left:

    st.caption(
        "© 2026 QORGAN AI"
    )

    st.caption(
        "Барлық құқық қорғалған"
    )


with footer_center:

    st.caption(
        "111 Keleshek School"
    )


with footer_right:

    st.caption(
        "111mektep@edu.kz"
    )