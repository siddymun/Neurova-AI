
import streamlit as st
import base64
import pandas as pd
from pathlib import Path
from predict_intent import predict_intent

st.set_page_config(
    page_title="Neurova AI| AI Voice Assistant",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
.mic-placeholder {
    height: 48px;
    width: 48px;
    margin-top: 8px;
    border: 1px solid #70418B;
    border-radius: 50%;
    background: linear-gradient(145deg, #24132F, #15101B);
    display: flex;
    align-items: center;
    justify-content: center;
    color: #E7C7FF;
    font-size: 21px;
}


@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

.stApp {
    background:
        radial-gradient(ellipse at 50% 0%, #381044 0%, transparent 36%),
        #08070B;
    color: #F8F3FF;
    font-family: 'Inter', sans-serif;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1050px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

[data-testid="stSidebar"] {
    background: #100D16;
    border-right: 1px solid #30203A;
}

.brand {
    color: #F5D9FF;
    font-size: 12px;
    letter-spacing: 4px;
    font-weight: 700;
}

.topline {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 26px;
}

.status {
    color: #C7A5D7;
    font-size: 12px;
    border: 1px solid #53305E;
    border-radius: 30px;
    padding: 7px 12px;
    background: #1B1021;
}

h1, h2, h3 {
    color: #FFF8FF !important;
}

.subtle {
    color: #A99BAF;
    font-size: 14px;
    line-height: 1.7;
}


.orb-wrap {
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 20px 0;
    background: transparent;
}


.orb-wrap {
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 20px 0;
    background: transparent;
}

.orb-image {
    width: 260px;
    height: 260px;
    display: block;
    object-fit: cover;

    /* Hide the square corners */
    border-radius: 50%;

    /* Rotate the image */
    animation: orb-rotate 12s linear infinite;
    transform-origin: center;

    filter: drop-shadow(0 0 24px rgba(168, 85, 247, 0.45));
}

@keyframes orb-rotate {
    from {
        transform: rotate(0deg);
    }
    to { transform: rotate(360deg);
    }
}

.orb {
    width: 170px;
    height: 170px;
    border-radius: 50%;
    background:
        radial-gradient(circle at 32% 25%,
            #FFE8FF 0%, #E8A7FF 9%,
            #BC48FF 26%, #7020D6 48%,
            #241044 70%, #0A0810 78%);
    box-shadow:
        0 0 25px #B832FF80,
        0 0 65px #9B28FF50,
        inset -12px -15px 30px #08030F;
    border: 1px solid #D9A1FF80;
}

.panel {
    background: linear-gradient(145deg, #17121D, #100D14);
    border: 1px solid #33253D;
    border-radius: 20px;
    padding: 22px;
    margin: 12px 0;
}

.feature {
    background: #151019;
    border: 1px solid #35263F;
    border-radius: 15px;
    padding: 17px;
    min-height: 105px;
}

.feature-icon {
    font-size: 22px;
    margin-bottom: 8px;
}

.feature-title {
    color: #F5E9FA;
    font-weight: 600;
    font-size: 14px;
}

.feature-desc {
    color: #A99BAF;
    font-size: 12px;
    margin-top: 5px;
}

div[data-testid="stChatMessage"] {
    background: #17121D;
    border: 1px solid #33253D;
    border-radius: 16px;
    padding: 14px;
}

/* Neurova AI — seamless message bar */
div[data-testid="stChatInput"] {
    background: #100B14 !important;
    border: 1px solid #33213D !important;
    border-radius: 18px !important;
    box-shadow: none !important;
    outline: none !important;
    transition: border-color 0.2s ease;
}

/* Remove the red focus outline */
div[data-testid="stChatInput"]:focus-within {
    border: 1px solid #654177 !important;
    box-shadow: 0 0 12px rgba(168, 85, 247, 0.10) !important;
    outline: none !important;
}

/* Text area */
div[data-testid="stChatInput"] textarea,
div[data-testid="stChatInput"] textarea:focus {
    background: transparent !important;
    color: #F8F3FF !important;
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
    -webkit-box-shadow: none !important;
}

/* Placeholder color — keep the lavender you like */
div[data-testid="stChatInput"] textarea::placeholder {
    color: #B9A6C8 !important;
    opacity: 1 !important;
}

/* Send button */
div[data-testid="stChatInput"] button {
    background: #21152A !important;
    color: #EAD5FF !important;
    border: none !important;
    box-shadow: none !important;
    border-radius: 10px !important;
}

div[data-testid="stChatInput"] button:hover {
    background: #382044 !important;
    color: #FFFFFF !important;
}

.stButton button {
    background: #21152A;
    color: #F9EFFF;
    border: 1px solid #654177;
    border-radius: 13px;
    min-height: 42px;
}

.stButton button:hover {
    border-color: #D18BFF;
    color: white;
}

div[data-testid="stAudioInput"] {
    border: 1px solid #654177;
    border-radius: 14px;
    padding: 12px;
    background: #151019;
}

hr {
    border-color: #302039;
}

footer {
    visibility: hidden;
}

/* Remove Streamlit's red focus and validation outlines */
div[data-testid="stChatInput"],
div[data-testid="stChatInput"]:focus,
div[data-testid="stChatInput"]:focus-within,
div[data-testid="stChatInput"] > div,
div[data-testid="stChatInput"] > div:focus-within {
    outline: none !important;
    box-shadow: none !important;
}

/* Keep the border purple while typing */
div[data-testid="stChatInput"]:focus-within {
    border: 1px solid #654177 !important;
}


</style>
""", unsafe_allow_html=True)

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hi! I'm NEUROVA. You can type a message below. "
                "We'll connect full voice recognition next."
            )
        }
    ]

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

# Sidebar

with st.sidebar:
    st.markdown("## ✦ Neurova AI")
    st.caption("YOUR AI COMPANION")
    st.divider()

    st.markdown("### Explore")
    page_view = st.radio(
        "Choose a page",
        ["AI Conversation", "Model Analytics"],
        index=0,
        key="page_view"
    )

    st.divider()

    st.markdown("### 🧠 Model Selection")

    model_choice = st.radio(
        "Choose your AI model",
        ["FNN", "LSTM", "Compare Both"],
        index=0
    )

    st.divider()
    # Check FNN model
    if Path("models/fnn_checkpoint_improved.pt").exists():
        st.success("FNN model available")
    else:
        st.warning("FNN model not found")

    # Check LSTM model
    if Path("models/lstm_checkpoint.pt").exists():
        st.success("LSTM model available")
    else:
        st.warning("LSTM model not found")

    st.divider()

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_prediction = None
        st.rerun()

# Model Analytics page
if page_view == "Model Analytics":
    st.markdown("""
    <div class="topline">
       <div class="brand">✦ NEUROVA AI</div>
       <div class="status">● MODEL ANALYTICS</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## 📊 Model Analytics Dashboard")
    st.markdown(
        '<p class="subtle">Evaluation results from the same 53-example test set '
        'using the 350-example intent dataset.</p>',
        unsafe_allow_html=True
    )

    # Headline evaluation metrics
    metric1, metric2, metric3 = st.columns(3)
    with metric1:
        st.metric("FNN Test Accuracy", "81.13%")
    with metric2:
        st.metric("LSTM Test Accuracy", "73.58%")
    with metric3:
        st.metric("FNN Advantage", "+7.55 pp")

    st.markdown("---")
    st.markdown("### Overall model comparison")

    accuracy_data = pd.DataFrame(
        {"Accuracy (%)": [81.13, 73.58]},
        index=["FNN", "LSTM"]
    )
    st.bar_chart(accuracy_data, y="Accuracy (%)", height=260)

    summary_data = pd.DataFrame([
        {"Metric": "Accuracy", "FNN": 0.81, "LSTM": 0.74},
        {"Metric": "Macro average F1", "FNN": 0.81, "LSTM": 0.73},
        {"Metric": "Weighted average F1", "FNN": 0.81, "LSTM": 0.73},
    ]).set_index("Metric")
    st.markdown("#### Evaluation summary")
    st.dataframe(
        summary_data.style.format("{:.2f}"),
        use_container_width=True
    )

    st.markdown("---")
    st.markdown("### Classification performance by intent")
    st.caption(
        "F1-score balances precision and recall. Values below are taken "
        "from the classification reports printed after model evaluation."
    )

    f1_data = pd.DataFrame([
        {"Intent": "Ask question", "FNN": 0.92, "LSTM": 0.74},
        {"Intent": "Exit", "FNN": 0.73, "LSTM": 0.33},
        {"Intent": "Greeting", "FNN": 0.73, "LSTM": 0.75},
        {"Intent": "Help", "FNN": 0.82, "LSTM": 0.71},
        {"Intent": "Save note", "FNN": 0.88, "LSTM": 0.80},
        {"Intent": "Set reminder", "FNN": 0.86, "LSTM": 1.00},
        {"Intent": "Thanks", "FNN": 0.77, "LSTM": 0.77},
    ]).set_index("Intent")
    st.bar_chart(f1_data, height=350)

    with st.expander("View precision, recall and F1-score values"):
        class_metrics = pd.DataFrame([
            ["Ask question", 1.00, 0.86, 0.92, 0.58, 1.00, 0.74],
            ["Exit", 1.00, 0.57, 0.73, 0.40, 0.29, 0.33],
            ["Greeting", 0.57, 1.00, 0.73, 0.75, 0.75, 0.75],
            ["Help", 0.78, 0.88, 0.82, 0.67, 0.75, 0.71],
            ["Save note", 0.88, 0.88, 0.88, 0.86, 0.75, 0.80],
            ["Set reminder", 0.86, 0.86, 0.86, 1.00, 1.00, 1.00],
            ["Thanks", 1.00, 0.62, 0.77, 1.00, 0.62, 0.77],
        ], columns=[
            "Intent", "FNN Precision", "FNN Recall", "FNN F1",
            "LSTM Precision", "LSTM Recall", "LSTM F1"
        ])
        st.dataframe(
            class_metrics.style.format({
                "FNN Precision": "{:.2f}", "FNN Recall": "{:.2f}",
                "FNN F1": "{:.2f}", "LSTM Precision": "{:.2f}",
                "LSTM Recall": "{:.2f}", "LSTM F1": "{:.2f}"
            }),
            use_container_width=True,
            hide_index=True
        )

    st.markdown("---")
    st.markdown("### Training loss")
    st.caption(
        "Loss values recorded during the 100 training epochs. "
        "Lower training loss means the model's error on its training data decreased."
    )
    loss_data = pd.DataFrame({
        "Epoch": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "FNN": [1.9157, 1.8644, 1.7905, 1.6837, 1.5372,
                1.3600, 1.1737, 0.9579, 0.7673, 0.5986],
        "LSTM": [1.8492, 1.7240, 1.5414, 1.3146, 1.0398,
                 0.7759, 0.5224, 0.3489, 0.2313, 0.1557]
    }).set_index("Epoch")
    st.line_chart(loss_data, height=300)

    st.markdown("---")
    st.markdown("### Live prediction confidence")
    st.caption(
        "Four example messages tested in Compare Both mode. "
        "These are sample prediction confidences, not overall accuracy."
    )
    confidence_data = pd.DataFrame({
        "Message": ["Can you help me?", "What is AI?",
                    "Thank you so much", "Remind me to study tomorrow"],
        "FNN": [60.87, 50.19, 65.64, 69.35],
        "LSTM": [97.70, 90.98, 92.77, 98.87]
    }).set_index("Message")
    st.bar_chart(confidence_data, height=300)

    st.markdown("---")
    st.markdown("### Key observations")
    st.markdown("""
    - **FNN achieved higher overall test accuracy:** 81.13% compared with 73.58% for LSTM.
    - **LSTM performed best on set_reminder:** its F1-score was 1.00.
    - **FNN performed better on exit:** its F1-score was 0.73 compared with 0.33 for LSTM.
    - Both models achieved an F1-score of 0.77 for the thanks intent.
    - In the four live examples shown above, both models predicted the same intent.
    """)
    st.info(
        "Interpretation note: test accuracy and classification metrics describe "
        "performance on the held-out test set. Training loss and prediction "
        "confidence are different measurements and should not be treated as accuracy."
    )

    st.markdown("---")
    st.markdown(
        '<p style="text-align:center;color:#786A80;font-size:11px;">'
        'NEUROVA AI · MODEL ANALYTICS · FNN + LSTM</p>',
        unsafe_allow_html=True
    )
    st.stop()

# Header
st.markdown("""
<div class="topline">
   <div class="brand">✦ NEUROVA AI</div>
    <div class="status">● AI WORKSPACE</div>
</div>
""", unsafe_allow_html=True)

st.markdown("## AI Voice Assistant")
st.markdown(
    '<p class="subtle">A more natural way to interact with artificial intelligence.</p>',
    unsafe_allow_html=True
)
# Neurova AI Orb Image

project_folder = Path(__file__).resolve().parent



orb_path = next(
    (p for p in (project_folder / "assests").glob("neurova-orb.*")),
    project_folder / "assests" / "neurova-orb.jpg"
)

if orb_path.exists():
    image_data = base64.b64encode(
        orb_path.read_bytes()
    ).decode("utf-8")

    st.markdown(
        f"""
        <div class="orb-wrap">
            <img
                class="orb-image"
                src="data:image/jpeg;base64,{image_data}"
                alt="Neurova AI particle sphere"
            />
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.warning("Image not found: assets/neurova-orb.jpg")

st.markdown(
    '<p style="text-align:center;color:#E7C8F6;'
    'font-size:16px;font-weight:600;">Your assistant is ready</p>',
    unsafe_allow_html=True
)
st.markdown(
    '<p style="text-align:center;color:#94859D;font-size:12px;">'
    'Ask a question or start a voice interaction</p>',
    unsafe_allow_html=True
)
# Feature cards
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="feature">
        <div class="feature-icon">🎙️</div>
        <div class="feature-title">Voice Assistant</div>
        <div class="feature-desc">Microphone interface</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="feature">
        <div class="feature-icon">🧠</div>
        <div class="feature-title">Deep Learning</div>
        <div class="feature-desc">FNN intent recognition</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="feature">
        <div class="feature-icon">✧</div>
        <div class="feature-title">Smart Interaction</div>
        <div class="feature-desc">Text and voice workflow</div>
    </div>
    """, unsafe_allow_html=True)


# Chat
st.markdown("### 💬 Conversation")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Message bar and microphone button
chat_col, mic_col = st.columns([10, 1])

with chat_col:
    user_input = st.chat_input("Message Neurova AI...")


with mic_col:
    st.markdown(
        """
        <div class="mic-placeholder">
            <span>🎙</span>
        </div>
        """,
        unsafe_allow_html=True
    )
# Process typed messages
if user_input:
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    try:
        if model_choice == "FNN":

            result = predict_intent(user_input)

            st.session_state.last_prediction = {
                "FNN": result
            }

            reply = (
                "**FNN Prediction**\n\n"
                f"Predicted intent: **{result['intent'].replace('_', ' ')}**\n\n"
                f"Confidence: **{result['confidence']:.2f}%**"
            )

        elif model_choice == "LSTM":

            from predict_lstm import predict_lstm

            intent, confidence = predict_lstm(user_input)

            st.session_state.last_prediction = {
                "LSTM": {
                    "intent": intent,
                    "confidence": confidence * 100
                }
            }

            reply = (
                "**LSTM Prediction**\n\n"
                f"Predicted intent: **{intent.replace('_', ' ')}**\n\n"
                f"Confidence: **{confidence * 100:.2f}%**"
            )

        else:

            from predict_lstm import predict_lstm

            fnn_result = predict_intent(user_input)
            lstm_intent, lstm_confidence = predict_lstm(user_input)

            st.session_state.last_prediction = {
                "FNN": fnn_result,
                "LSTM": {
                    "intent": lstm_intent,
                    "confidence": lstm_confidence * 100
                }
            }
            reply = (
                "**Model Comparison**\n\n"
                "**FNN**\n\n"
                f"Intent: **{fnn_result['intent'].replace('_', ' ')}**\n\n"
                f"Confidence: **{fnn_result['confidence']:.2f}%**\n\n"
                "---\n\n"
                "**LSTM**\n\n"
                f"Intent: **{lstm_intent.replace('_', ' ')}**\n\n"
                f"Confidence: **{lstm_confidence * 100:.2f}%**"
            )

            if fnn_result["intent"] == lstm_intent:
                reply += "\n\n**Agreement:** Both models predicted the same intent."
            else:
                reply += "\n\n**Agreement:** The models predicted different intents."

    except FileNotFoundError as error:
        reply = f"Model checkpoint not found: {error}"

    except Exception as error:
        reply = f"Prediction error: {error}"

    st.session_state.messages.append(
        {"role": "assistant", "content": reply}
    )

    st.rerun()

# Audio recording feedback

# Footer
st.markdown("---")
st.markdown(
    '<p style="text-align:center;color:#786A80;font-size:11px;">'
    'NOVA AI · Deep Learning · Intent Recognition</p>',
    unsafe_allow_html=True
)
