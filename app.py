import streamlit as st
import joblib

st.set_page_config(
    page_title="Roman Urdu Sentiment Analyzer",
    page_icon="🌍",
    layout="centered"
)

# Custom CSS
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .title {
        text-align: center;
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .subtitle {
        text-align: center;
        color: #6c757d;
        font-size: 1rem;
        margin-top: 0;
    }
    .developer {
        text-align: center;
        color: #764ba2;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 1rem 0;
    }
    .positive-box {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        border: 2px solid #28a745;
    }
    .negative-box {
        background: linear-gradient(135deg, #f8d7da, #f5c6cb);
        border: 2px solid #dc3545;
    }
    .result-emoji {
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }
    .result-label {
        font-size: 1.8rem;
        font-weight: 800;
        margin: 0;
    }
    .result-confidence {
        font-size: 1rem;
        color: #495057;
        margin-top: 0.3rem;
    }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 8px;
        font-size: 1rem;
        font-weight: 600;
        padding: 0.6rem 2rem;
        width: 100%;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        opacity: 0.9;
        transform: translateY(-1px);
    }
    .footer {
        text-align: center;
        color: #adb5bd;
        font-size: 0.8rem;
        margin-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model = joblib.load("sentiment_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# Header
st.markdown('<p class="title">🌍 Roman Urdu Sentiment Analyzer</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Analyze sentiment of Roman Urdu & English text using Machine Learning</p>', unsafe_allow_html=True)
st.markdown('<p class="developer">👨‍💻 Developed by Muhammad Arif</p>', unsafe_allow_html=True)
st.divider()

# Examples
st.markdown("**💡 Try these examples:**")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("yeh movie achi thi"):
        st.session_state.example = "yeh movie achi thi"
with col2:
    if st.button("bohat bekar service thi"):
        st.session_state.example = "bohat bekar service thi"
with col3:
    if st.button("zabardast experience tha"):
        st.session_state.example = "zabardast experience tha"

# Input
default_text = st.session_state.get("example", "")
user_input = st.text_area(
    "✍️ Enter your text here:",
    value=default_text,
    placeholder="e.g. yeh movie bohat achi thi...",
    height=130
)

# Analyze button
if st.button("🔍 Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text first!")
    else:
        with st.spinner("Analyzing sentiment..."):
            text_tfidf  = vectorizer.transform([user_input])
            prediction  = model.predict(text_tfidf)[0]
            probability = model.predict_proba(text_tfidf)[0]
            confidence  = probability[prediction] * 100

        st.divider()

        # Result box
        if prediction == 1:
            st.markdown(f"""
                <div class="result-box positive-box">
                    <div class="result-emoji">😊</div>
                    <p class="result-label" style="color:#155724;">POSITIVE</p>
                    <p class="result-confidence">{confidence:.1f}% confidence</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="result-box negative-box">
                    <div class="result-emoji">😞</div>
                    <p class="result-label" style="color:#721c24;">NEGATIVE</p>
                    <p class="result-confidence">{confidence:.1f}% confidence</p>
                </div>
            """, unsafe_allow_html=True)

        # Progress bars
        st.markdown("**📊 Confidence Breakdown:**")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("😊 Positive", f"{probability[1]*100:.1f}%")
            st.progress(float(probability[1]))
        with col2:
            st.metric("😞 Negative", f"{probability[0]*100:.1f}%")
            st.progress(float(probability[0]))

        # Input summary
        st.info(f"📝 Analyzed: *\"{user_input[:80]}{'...' if len(user_input) > 80 else ''}\"*")

# Footer
st.divider()
st.markdown("""
    <div class="footer">
        Built with TF-IDF + Logistic Regression | Trained on 11,018 Roman Urdu Reviews<br>
        Made with ❤️ by <b>Muhammad Arif</b> | Sukkur IBA University
    </div>
""", unsafe_allow_html=True)