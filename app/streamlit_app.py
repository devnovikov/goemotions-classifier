"""
Streamlit UI for GoEmotions Multi-Label Emotion Classifier.

Target versions:
- streamlit>=1.30.0
- plotly>=5.18.0
- python>=3.11
"""

import os
import sys
from pathlib import Path
from typing import Optional

import plotly.graph_objects as go
import requests
import streamlit as st

# Add parent directory to path for imports (for local development)
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import from src.constants, fallback to embedded constants for Streamlit Cloud
try:
    from src.constants import DEFAULT_THRESHOLD, EMOTION_COLORS, EMOTION_EMOJIS
except ImportError:
    # Embedded constants for standalone deployment (Streamlit Cloud)
    DEFAULT_THRESHOLD = 0.35

    EMOTION_COLORS = {
        "joy": "#FFD700", "love": "#FF6B9D", "excitement": "#FF6B6B",
        "gratitude": "#26DE81", "admiration": "#A55EEA", "amusement": "#FF9F43",
        "optimism": "#FED330", "pride": "#9B59B6", "relief": "#7BED9F",
        "approval": "#4ECDC4", "caring": "#FF9FF3", "desire": "#E84393",
        "anger": "#EE5A5A", "sadness": "#74B9FF", "fear": "#2D3436",
        "disappointment": "#636E72", "annoyance": "#FC8181", "disgust": "#6C5CE7",
        "grief": "#4A5568", "remorse": "#81ECEC", "nervousness": "#FDCB6E",
        "confusion": "#A29BFE", "curiosity": "#00CEC9", "realization": "#55EFC4",
        "surprise": "#FD79A8", "embarrassment": "#FAB1A0",
        "neutral": "#95A5A6", "disapproval": "#B2BEC3",
    }

    EMOTION_EMOJIS = {
        "admiration": "🤩", "amusement": "😄", "anger": "😠", "annoyance": "😒",
        "approval": "👍", "caring": "🤗", "confusion": "😕", "curiosity": "🤔",
        "desire": "😍", "disappointment": "😞", "disapproval": "👎", "disgust": "🤢",
        "embarrassment": "😳", "excitement": "🤩", "fear": "😨", "gratitude": "🙏",
        "grief": "😢", "joy": "😊", "love": "❤️", "nervousness": "😰",
        "optimism": "🌟", "pride": "😌", "realization": "💡", "relief": "😌",
        "remorse": "😔", "sadness": "😢", "surprise": "😲", "neutral": "😐",
    }

# ============================================================================
# Configuration
# ============================================================================

# Support both environment variables and Streamlit secrets (for Streamlit Cloud)
def get_api_url() -> str:
    """Get API URL from secrets, env var, or default."""
    # 1. Try Streamlit secrets (for Streamlit Cloud)
    try:
        if hasattr(st, "secrets") and "API_URL" in st.secrets:
            return st.secrets["API_URL"]
    except Exception:
        pass
    # 2. Try environment variable
    # 3. Default to localhost
    return os.environ.get("API_URL", "http://localhost:8000")


API_URL = get_api_url()

st.set_page_config(
    page_title="GoEmotions Classifier",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# Custom CSS - Light Theme (Clean Professional)
# ============================================================================


def apply_custom_css() -> None:
    """Apply custom light theme styling for Streamlit 1.30+."""
    st.markdown(
        """
        <style>
        /* Light theme - Clean Professional */
        .stApp {
            background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 100%);
        }

        /* Sidebar - light theme */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #ffffff 0%, #f8f9fa 100%);
            border-right: 1px solid #e0e0e0;
        }

        section[data-testid="stSidebar"] .stMarkdown {
            color: #333333;
        }

        /* Metric cards - light */
        .metric-card {
            background: #ffffff;
            border: 1px solid #e0e0e0;
            border-radius: 12px;
            padding: 16px;
            margin: 8px 0;
            text-align: center;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        }

        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #2563eb;
        }

        .metric-label {
            font-size: 12px;
            color: #6b7280;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        /* Status indicator */
        .status-online {
            display: inline-block;
            width: 10px;
            height: 10px;
            background: #22c55e;
            border-radius: 50%;
            margin-right: 8px;
            animation: pulse 2s infinite;
        }

        .status-offline {
            display: inline-block;
            width: 10px;
            height: 10px;
            background: #ef4444;
            border-radius: 50%;
            margin-right: 8px;
        }

        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }

        /* Emotion tag styling */
        .emotion-tag {
            display: inline-block;
            padding: 6px 14px;
            margin: 4px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: 500;
            color: white;
            text-shadow: 0 1px 2px rgba(0,0,0,0.2);
        }

        /* Success/detected emotions box */
        .detected-emotions {
            background: rgba(34, 197, 94, 0.1);
            border: 1px solid rgba(34, 197, 94, 0.3);
            border-radius: 12px;
            padding: 16px;
            margin: 16px 0;
        }

        /* Text area - Streamlit 1.30+ selectors */
        div[data-testid="stTextArea"] textarea {
            background: #ffffff !important;
            border: 1px solid #d1d5db !important;
            border-radius: 8px !important;
            color: #1f2937 !important;
        }

        div[data-testid="stTextArea"] textarea:focus {
            border-color: #2563eb !important;
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1) !important;
        }

        /* Button styling - Streamlit 1.30+ */
        div[data-testid="stButton"] > button {
            background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 10px 24px;
            font-weight: 600;
            transition: all 0.2s ease;
            box-shadow: 0 2px 8px rgba(37, 99, 235, 0.25);
        }

        div[data-testid="stButton"] > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.35);
        }

        /* Headings - light */
        h1 {
            color: #1f2937 !important;
        }

        h2, h3 {
            color: #1f2937 !important;
        }

        /* Info text */
        .info-text {
            color: #6b7280;
            font-size: 14px;
        }

        /* Score list item */
        .score-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 12px;
            margin: 4px 0;
            background: #f9fafb;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
        }

        .score-value {
            color: #2563eb;
            font-weight: 600;
        }

        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}

        /* Expander - Streamlit 1.30+ */
        div[data-testid="stExpander"] {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
        }

        div[data-testid="stExpander"] summary {
            background: #f9fafb !important;
            border-radius: 8px !important;
        }

        /* Slider styling for light theme */
        div[data-testid="stSlider"] label {
            color: #374151 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================================
# API Functions
# ============================================================================


def check_api_health() -> tuple[bool, Optional[dict]]:
    """Check if the API is available and healthy."""
    try:
        response = requests.get(f"{API_URL}/", timeout=5)
        if response.status_code == 200:
            return True, response.json()
        return False, None
    except requests.exceptions.RequestException:
        return False, None


def get_model_info() -> Optional[dict]:
    """Get model information from the API."""
    try:
        response = requests.get(f"{API_URL}/model/info", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except requests.exceptions.RequestException:
        return None


def classify_text(text: str, threshold: float) -> Optional[dict]:
    """Send text to API for classification."""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={"text": text, "threshold": threshold},
            timeout=30,
        )
        if response.status_code == 200:
            return response.json()
        return None
    except requests.exceptions.RequestException:
        return None


# ============================================================================
# Chart Creation (Plotly >= 5.18.0)
# ============================================================================


def create_emotion_chart(scores: dict, threshold: float) -> go.Figure:
    """Create a horizontal bar chart for emotion scores."""
    sorted_scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

    emotions = list(sorted_scores.keys())
    values = list(sorted_scores.values())
    colors = [EMOTION_COLORS.get(e, "#666666") for e in emotions]

    max_v = max(values) if values else 1.0
    x_max = max(1.0, max_v * 1.15, threshold * 1.15)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=emotions,
            x=values,
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"{v:.1%}" for v in values],
            textposition="outside",
            textfont=dict(color="#374151", size=11),
            hovertemplate="<b>%{y}</b><br>Score: %{x:.2%}<extra></extra>",
            cliponaxis=False,
        )
    )

    fig.add_vline(
        x=threshold,
        line=dict(color="#FF6B6B", width=2, dash="dash"),
        annotation_text=f"Threshold ({threshold:.0%})",
        annotation_position="top",
        annotation=dict(font=dict(color="#FF6B6B", size=12)),
    )

    fig.update_layout(
        title=dict(
            text="Emotion Confidence Scores",
            font=dict(color="#1f2937", size=16),
        ),
        xaxis=dict(
            title="Confidence Score",
            tickformat=".0%",
            range=[0, x_max],
            gridcolor="rgba(0, 0, 0, 0.1)",
            tickfont=dict(color="#6b7280"),
            title_font=dict(color="#374151"),
        ),
        yaxis=dict(
            tickfont=dict(color="#374151"),
            autorange="reversed",
        ),
        plot_bgcolor="#ffffff",
        paper_bgcolor="rgba(0,0,0,0)",
        height=max(320, len(emotions) * 34),
        margin=dict(l=10, r=120, t=55, b=40),
        showlegend=False,
    )

    return fig


# ============================================================================
# Sidebar
# ============================================================================


def render_sidebar() -> None:
    """Render the sidebar with model information."""
    with st.sidebar:
        st.markdown("## Model Information")

        api_healthy, _health_info = check_api_health()

        if api_healthy:
            st.markdown(
                '<p><span class="status-online"></span> API Online</p>',
                unsafe_allow_html=True,
            )

            model_info = get_model_info()

            if model_info:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Model</div>
                        <div class="metric-value" style="font-size: 16px;">{model_info.get('name', 'Unknown')}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">Type</div>
                            <div class="metric-value" style="font-size: 14px;">{model_info.get('type', 'Unknown')}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with col2:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">Version</div>
                            <div class="metric-value" style="font-size: 14px;">{model_info.get('version', 'Unknown')}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                trained_at = model_info.get("trained_at", "Unknown")
                if trained_at and trained_at != "Unknown":
                    trained_at = str(trained_at).split("T")[0]
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Trained</div>
                        <div class="metric-value" style="font-size: 14px;">{trained_at}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown("### Performance Metrics")
                metrics = model_info.get("metrics", {})

                if metrics:
                    col1, col2 = st.columns(2)
                    with col1:
                        micro_f1 = metrics.get("micro_f1", 0)
                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <div class="metric-label">Micro F1</div>
                                <div class="metric-value">{micro_f1:.3f}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    with col2:
                        macro_f1 = metrics.get("macro_f1", 0)
                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <div class="metric-label">Macro F1</div>
                                <div class="metric-value">{macro_f1:.3f}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    hamming = metrics.get("hamming_loss", 0)
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">Hamming Loss</div>
                            <div class="metric-value">{hamming:.4f}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.info("No metrics available")
            else:
                st.warning("Could not load model info")
        else:
            st.markdown(
                '<p><span class="status-offline"></span> API Offline</p>',
                unsafe_allow_html=True,
            )
            st.error("Cannot connect to API. Make sure the server is running.")
            st.code("uvicorn src.predict.predict:app --reload", language="bash")

        st.markdown("---")
        st.markdown("### About")
        st.markdown(
            """
            <div class="info-text">
            This classifier uses the <a href="https://github.com/google-research/google-research/tree/master/goemotions"
            target="_blank" style="color: #2563eb;">GoEmotions</a> dataset by Google Research
            to detect 28 emotion categories in text.
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================================
# Results Display
# ============================================================================


def render_results(response: dict) -> None:
    """Render classification results."""
    st.markdown("---")
    st.markdown("## Results")

    labels = response.get("labels", []) or []
    scores = response.get("scores", {}) or {}
    threshold = float(response.get("threshold", DEFAULT_THRESHOLD))
    inference_time = float(response.get("inference_time_ms", 0) or 0)
    model_type = str(response.get("model_type", "unknown"))

    if labels:
        st.markdown("### Detected Emotions")
        emotions_html = '<div class="detected-emotions">'
        for label in labels:
            color = EMOTION_COLORS.get(label, "#666666")
            emoji = EMOTION_EMOJIS.get(label, "")
            score = float(scores.get(label, 0) or 0)
            emotions_html += (
                f'<span class="emotion-tag" style="background: {color};">'
                f"{emoji} {label.capitalize()} ({score:.0%})</span>"
            )
        emotions_html += "</div>"
        st.markdown(emotions_html, unsafe_allow_html=True)
    else:
        st.info("No emotions detected above the threshold.")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### Confidence Scores")
        for emotion, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            score = float(score or 0)
            emoji = EMOTION_EMOJIS.get(emotion, "")
            color = EMOTION_COLORS.get(emotion, "#666666")
            weight = "700" if score >= threshold else "400"

            st.markdown(
                f"""
                <div class="score-item">
                    <span>{emoji} {str(emotion).capitalize()}</span>
                    <span class="score-value" style="color: {color}; font-weight: {weight};">
                        {score:.1%}
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with col2:
        if scores:
            fig = create_emotion_chart(scores, threshold)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Inference Time", f"{inference_time:.1f} ms")
    with c2:
        st.metric("Model Type", model_type.upper())
    with c3:
        st.metric("Threshold", f"{threshold:.0%}")


# ============================================================================
# Main Application
# ============================================================================


def main() -> None:
    apply_custom_css()
    render_sidebar()

    st.markdown(
        """
        # 🎭 GoEmotions Multi-Label Emotion Classifier

        <p class="info-text">
        Analyze text and detect up to <b>28 different emotions</b> using state-of-the-art
        transformer models trained on Google's GoEmotions dataset. This multi-label classifier
        can identify multiple emotions present in a single text.
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Enter Text to Analyze")

    # --- Fix for Streamlit 1.30+ session_state limitation:
    # Apply pending text BEFORE widget instantiation.
    if "text_input" not in st.session_state:
        st.session_state["text_input"] = ""
    if "pending_text" in st.session_state:
        st.session_state["text_input"] = st.session_state.pop("pending_text")

    st.text_area(
        "Text",
        placeholder="Type or paste your text here...\n\nExample: I'm so excited about my promotion but also nervous about the new responsibilities!",
        height=150,
        label_visibility="collapsed",
        key="text_input",
    )

    col1, col2 = st.columns([3, 1])

    with col1:
        threshold = st.slider(
            "Classification Threshold",
            min_value=0.0,
            max_value=1.0,
            value=float(DEFAULT_THRESHOLD),
            step=0.05,
            help="Emotions with confidence scores above this threshold will be detected.",
        )

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        classify_button = st.button("🔍 Classify", use_container_width=True)

    if classify_button:
        text_input = (st.session_state.get("text_input") or "").strip()
        if not text_input:
            st.warning("Please enter some text to classify.")
        else:
            with st.spinner("Analyzing emotions..."):
                response = classify_text(text_input, threshold)
                if response:
                    render_results(response)
                else:
                    st.error("Failed to get prediction. Make sure the API is running and accessible.")

    with st.expander("📝 Try Example Texts"):
        examples = [
            "I'm so happy and grateful for all your help! You're amazing!",
            "This is absolutely terrible. I'm furious about what happened.",
            "I don't know what to think about this situation. It's confusing.",
            "I'm nervous about the interview tomorrow but also excited for the opportunity.",
            "That movie was surprisingly good! I didn't expect to enjoy it so much.",
        ]

        for i, example in enumerate(examples, start=1):
            if st.button(f"Example {i}", key=f"example_{i}"):
                # IMPORTANT: do NOT assign to st.session_state["text_input"] here,
                # because the widget is already instantiated in this run.
                st.session_state["pending_text"] = example
                st.rerun()


if __name__ == "__main__":
    main()
