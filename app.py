
import os
import json
import time
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path

# -----------------------
# App metadata & disclaimer
# -----------------------
APP_NAME = "NyaySathi — Legal Aid Chatbot (Info-Only)"
DISCLAIMER = (
    "This chatbot provides **general legal information** (India-first) and is **not legal advice**. "
    "Laws vary by state and change over time. For any specific case, consult a qualified lawyer or contact "
    "your District Legal Services Authority (DLSA). If you are in danger or facing an emergency, dial 112 immediately."
)

# -----------------------
# Load knowledge base and build retriever (TF-IDF)
# -----------------------
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

KB_PATH = "kb/legal_kb.csv"

@st.cache_resource(show_spinner=False)
def load_kb_and_vectorizer():
    kb = pd.read_csv(KB_PATH)
    def _normalize(text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return re.sub(r"\\s+", " ", text)

    kb["blob"] = (kb["topic"].fillna("") + " " + kb["question"].fillna("") + " " + kb["answer"].fillna("")).map(_normalize)
    vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=1, max_features=20000)
    X = vectorizer.fit_transform(kb["blob"].tolist())
    return kb, vectorizer, X

kb, vectorizer, X = load_kb_and_vectorizer()

FORBIDDEN = [
    "help me evade law", "illegal", "forged document", "fake id", "violence", "harm someone",
    "bribe", "terror", "sell drugs"
]

def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\\s]", " ", text)
    return re.sub(r"\\s+", " ", text)

def is_disallowed(q: str) -> bool:
    qn = _normalize(q)
    return any(term in qn for term in FORBIDDEN)

def retrieve(query: str, topk: int = 3):
    qn = _normalize(query)
    qv = vectorizer.transform([qn])
    sims = cosine_similarity(qv, X).ravel()
    idx = np.argsort(-sims)[:topk]
    rows = kb.iloc[idx].copy()
    rows["score"] = sims[idx]
    return rows

def local_answer(query: str):
    if is_disallowed(query):
        return {
            "answer": (
                "I can’t assist with illegal or harmful activity. If you or someone is in danger, "
                "call local emergency services (India: 112)."
            ),
            "sources": [],
            "score": 0.0
        }
    rows = retrieve(query, topk=3)
    best = rows.iloc[0]
    conf = float(best["score"])
    if conf < 0.1:
        prefix = (
            "I might not have an exact match. Here are general steps:\\n"
            "• Collect documents and evidence.\\n"
            "• Approach your local police/authority or District Legal Services Authority (DLSA).\\n"
            "• Keep written acknowledgments.\\n\\n"
        )
    else:
        prefix = ""
    ans = (
        prefix
        + str(best["answer"])
        + "\\n\\n**Safety**: This is general information, not legal advice. Consider contacting your DLSA/lawyer."
    )
    sources = rows[["topic","question","source","score"]].to_dict(orient="records")
    return {"answer": ans, "sources": sources, "score": conf}

# -----------------------
# Optional: LLM polish via OpenAI (fully optional)
# -----------------------
def try_llm_polish(text: str, user_query: str, system_tone="Be clear, brief, and neutral. Do not invent facts."):
    use_polish = st.session_state.get("use_llm_polish", False)
    api_key = st.session_state.get("OPENAI_API_KEY")
    if not use_polish or not api_key:
        return text

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        messages = [
            {"role": "system", "content": system_tone + " Keep legal safety disclaimers intact."},
            {"role": "user", "content": f"User question: {user_query}\\n\\nGrounded draft answer (do not change facts):\\n{text}\\n\\nRewrite more clearly but keep meaning."}
        ]
        # Use a lightweight model name you have access to; you can change this later
        resp = client.chat.completions.create(model="gpt-4o-mini", messages=messages, temperature=0.2)
        return resp.choices[0].message.content.strip()
    except Exception as e:
        st.info(f"LLM polish skipped ({e}).")
        return text

# -----------------------
# UI
# -----------------------
st.set_page_config(page_title=APP_NAME, page_icon="⚖️", layout="wide")

st.title("⚖️ NyaySathi — Legal Aid Chatbot")
st.markdown(f"> {DISCLAIMER}")

with st.sidebar:
    st.header("Settings")
    st.toggle("Show sources", value=True, key="show_sources")
    st.toggle("Use optional LLM polish (needs OpenAI key)", value=False, key="use_llm_polish")

    st.markdown("**Optional — Add OpenAI API Key**")
    # Prefer Streamlit secrets in production; allow manual input for local
    default_key = st.secrets.get("OPENAI_API_KEY", "")
    key_input = st.text_input("OPENAI_API_KEY", value=default_key if default_key else "", type="password", placeholder="sk-...")
    if key_input:
        st.session_state["OPENAI_API_KEY"] = key_input

    st.markdown("---")
    st.subheader("Extend knowledge")
    uploaded = st.file_uploader("Upload extra FAQs (CSV with columns: topic,question,answer,source)", type=["csv"])
    if uploaded:
        try:
            extra = pd.read_csv(uploaded)
            req_cols = {"topic","question","answer"}
            if not req_cols.issubset(set(map(str.lower, extra.columns))):
                st.warning("CSV must include at least: topic, question, answer (case-insensitive).")
            else:
                # Normalize columns
                colmap = {c: c.lower() for c in extra.columns}
                extra = extra.rename(columns=colmap)
                if "source" not in extra.columns:
                    extra["source"] = "User-provided"
                # Append & rebuild vectorizer
                global kb, vectorizer, X
                kb = pd.concat([kb, extra[["topic","question","answer","source"]]], ignore_index=True)
                kb["blob"] = (kb["topic"].fillna("") + " " + kb["question"].fillna("") + " " + kb["answer"].fillna("")).map(_normalize)
                vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=1, max_features=20000)
                X = vectorizer.fit_transform(kb["blob"].tolist())
                st.success(f"Loaded {len(extra)} extra rows. Knowledge base now has {len(kb)} entries.")
        except Exception as e:
            st.error(f"Could not load CSV: {e}")

    st.markdown("---")
    st.caption("Quick topics")
    quick = st.selectbox(
        "Try a common query",
        ["(Choose one)", "Who is eligible for free legal aid?", "How do I file an FIR?", "Domestic violence help", "Consumer complaint process", "Report cybercrime", "Cheque bounce next steps", "Landlord not returning deposit"]
    )
    if quick and quick != "(Choose one)":
        st.session_state["draft"] = quick

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role":"assistant","content":"Hello! Ask me a legal-aid question in simple words. I’ll do my best to help with general information."}]

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

draft = st.session_state.get("draft","")
user_input = st.chat_input("Type your question...", key="chat_input", max_chars=500)

if user_input:
    st.session_state.messages.append({"role":"user","content":user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            base = local_answer(user_input)
            text = base["answer"]
            text = try_llm_polish(text, user_input)
            st.markdown(text)

            if st.session_state.get("show_sources", True):
                with st.expander("Why this answer? (top matches)"):
                    st.write(pd.DataFrame(base["sources"]))

    # lightweight feedback
    fb_cols = st.columns(2)
    if fb_cols[0].button("👍 Helpful", use_container_width=True):
        pass
    if fb_cols[1].button("👎 Not helpful", use_container_width=True):
        pass

# Footer help box
st.info("Need urgent help? Dial **112** (emergency). For free legal aid, contact your **District Legal Services Authority (DLSA)**.")
