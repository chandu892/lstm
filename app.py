import streamlit as st
import numpy as np
import pickle

from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ==================================================
# Page Config
# ==================================================

st.set_page_config(
    page_title="LSTM Next Word Predictor",
    page_icon="📖",
    layout="wide"
)

# ==================================================
# Custom CSS
# ==================================================

st.markdown("""
<style>
.main-title{
    font-size:40px;
    font-weight:bold;
    text-align:center;
    color:#4F8BF9;
}
.sub-title{
    text-align:center;
    font-size:18px;
    color:gray;
}
.result-box{
    padding:20px;
    border-radius:10px;
    background-color:#F0F2F6;
    font-size:22px;
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# Paths
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "lstm_model.keras"
TOKENIZER_PATH = BASE_DIR / "models" / "tokenizer.pkl"
MAXLEN_PATH = BASE_DIR / "models" / "max_seq_len.pkl"

# ==================================================
# Load Artifacts
# ==================================================

@st.cache_resource
def load_artifacts():

    model = load_model(MODEL_PATH)

    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)

    with open(MAXLEN_PATH, "rb") as f:
        max_seq_len = pickle.load(f)

    return model, tokenizer, max_seq_len

model, tokenizer, max_seq_len = load_artifacts()

index_word = {v: k for k, v in tokenizer.word_index.items()}

# ==================================================
# Temperature Sampling
# ==================================================

def sample_with_temperature(preds, temperature=0.8):

    preds = np.asarray(preds).astype("float64")

    preds = np.log(preds + 1e-10) / temperature

    exp_preds = np.exp(preds)

    preds = exp_preds / np.sum(exp_preds)

    return np.random.choice(
        len(preds),
        p=preds
    )

# ==================================================
# Generate Text
# ==================================================

def generate_text(seed_text, num_words, temperature):

    output = seed_text

    for _ in range(num_words):

        token_list = tokenizer.texts_to_sequences(
            [output]
        )[0]

        token_list = pad_sequences(
            [token_list],
            maxlen=max_seq_len - 1,
            padding="pre"
        )

        prediction = model.predict(
            token_list,
            verbose=0
        )[0]

        predicted_index = sample_with_temperature(
            prediction,
            temperature
        )

        next_word = index_word.get(
            predicted_index,
            ""
        )

        output += " " + next_word

    return output

# ==================================================
# Sidebar
# ==================================================

with st.sidebar:

    st.header("⚙️ Generation Settings")

    num_words = st.slider(
        "Words to Generate",
        min_value=1,
        max_value=20,
        value=5
    )

    temperature = st.slider(
        "Creativity (Temperature)",
        min_value=0.1,
        max_value=1.5,
        value=0.7,
        step=0.1
    )

    st.markdown("---")

    st.info("""
**Temperature Guide**

0.3 → Conservative

0.7 → Balanced

1.2 → Creative
""")

# ==================================================
# Main UI
# ==================================================

st.markdown(
    '<p class="main-title">📖 LSTM Next Word Predictor</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="sub-title">Generate Shakespeare-style text using an LSTM Neural Network</p>',
    unsafe_allow_html=True
)

st.markdown("---")

col1, col2 = st.columns([3,1])

with col1:

    seed_text = st.text_area(
        "Enter Starting Text",
        value="To be or not to",
        height=120
    )

with col2:

    st.metric(
        "Vocabulary Size",
        len(tokenizer.word_index)
    )

    st.metric(
        "Sequence Length",
        max_seq_len
    )

# ==================================================
# Generate Button
# ==================================================

if st.button("🚀 Generate Text", use_container_width=True):

    if len(seed_text.strip()) == 0:

        st.warning("Please enter some text.")

    else:

        with st.spinner("Generating text..."):

            result = generate_text(
                seed_text,
                num_words,
                temperature
            )

        st.success("Text Generated Successfully!")

        st.markdown("### Generated Text")

        st.markdown(
            f"""
            <div class="result-box">
            {result}
            </div>
            """,
            unsafe_allow_html=True
        )

# ==================================================
# Footer
# ==================================================

st.markdown("---")

with st.expander("📚 About This Project"):

    st.write("""
    **Model:** LSTM (Long Short-Term Memory)

    **Dataset:** Shakespeare Text Corpus

    **Task:** Next Word Prediction

    **Frameworks Used:**
    - TensorFlow/Keras
    - Streamlit
    - NumPy
    """)

st.caption("Developed with ❤️ using TensorFlow and Streamlit")