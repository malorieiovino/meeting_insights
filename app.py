import os
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─── Load our tiny demo CSV ────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_ami_dataset():
    """
    Read the five-column CSV we committed:
      meeting_id,speaker,begin_time,end_time,text_clean
    """
    path = os.path.join(os.path.dirname(__file__), "data", "sample_meeting.csv")
    df   = pd.read_csv(path)
    # Convert each row back into a dict for compatibility
    return df.to_dict(orient="records")

# ─── Helpers to slice out one meeting ──────────────────────────────────────
@st.cache_data
def get_meeting_segments(meeting_id: str):
    records = load_ami_dataset()
    segs = [r for r in records if r["meeting_id"] == meeting_id]
    if not segs:
        st.warning(f"⚠️ No segments found for meeting_id={meeting_id}")
    return segs

# ─── Extractive summarization utilities ──────────────────────────────────
action_keywords = ['will','need to','going to','must','should','action','task','by','due']

def extract_features(segments):
    texts = [s["text_clean"] for s in segments]
    vect  = TfidfVectorizer(stop_words="english")
    tfidf = vect.fit_transform(texts).toarray()
    doc_vec = tfidf.mean(axis=0).reshape(1, -1)
    sims = cosine_similarity(tfidf, doc_vec).flatten()

    feats = []
    for i, seg in enumerate(segments):
        feats.append({
            "idx":        i,
            "length":     len(seg["text_clean"].split()),
            "sim":        sims[i],
            "contains_a": any(k in seg["text_clean"].lower() for k in action_keywords),
            "is_new_spkr": i == 0 or seg["speaker"] != segments[i-1]["speaker"]
        })
    return feats

def generate_extractive_summary(segments, top_n):
    """
    Score each segment by a mixture of similarity-to-document, length,
    presence of action-keywords, and speaker-change, then pick top_n.
    """
    if not segments:
        return []
    feats = extract_features(segments)
    scores = [
        (
            f["idx"],
            0.4*f["sim"]
          + 0.3*min(f["length"]/20, 1.0)
          + 0.2*int(f["contains_a"])
          + 0.1*int(f["is_new_spkr"])
        )
        for f in feats
    ]
    ranked = sorted(scores, key=lambda x: x[1], reverse=True)
    top_idxs = sorted(idx for idx, _ in ranked[:top_n])
    return [segments[i] for i in top_idxs]

# ─── Clustering into topics ────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def cluster_into_topics(segments, n_topics):
    if not segments:
        return []
    # Embed & cluster
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [s["text_clean"] for s in segments]
    emb   = model.encode(texts, convert_to_tensor=True).cpu().numpy()
    km    = KMeans(n_clusters=n_topics, random_state=42)
    labels= km.fit_predict(emb)

    # Group into topic buckets
    topics = []
    for t in range(n_topics):
        topics.append([seg for seg, lab in zip(segments, labels) if lab == t])
    return topics

def summarize_meeting(segments, n_topics, top_n):
    topics = cluster_into_topics(segments, n_topics)
    rows = []
    for t_idx, topic in enumerate(topics):
        summary = generate_extractive_summary(topic, top_n)
        for seg in summary:
            rows.append({
                "Topic":     t_idx,
                "Speaker":   seg["speaker"],
                "Utterance": seg["text_clean"][:80] + ("…" if len(seg["text_clean"])>80 else "")
            })
    if not rows:
        st.error("No summary could be generated (no data).")
    return pd.DataFrame(rows)

# ─── Streamlit UI ─────────────────────────────────────────────────────────
st.set_page_config(page_title="AMI Meeting Summarizer", layout="wide")
st.title("AMI IHM Meeting Summarization Demo")

# Sidebar: choose from the small list of meeting_ids
st.sidebar.header("⚙️ Parameters")
meeting_ids = sorted({r["meeting_id"] for r in load_ami_dataset()})
sel = st.sidebar.selectbox("Select meeting_id", meeting_ids)
n_topics = st.sidebar.slider("Number of topics (clusters)", 2, 10, 5)
top_n    = st.sidebar.slider("Utterances per topic", 1, 5, 3)

# Show raw data
with st.expander("▶️ View raw segments"):
    segs   = get_meeting_segments(sel)
    if segs:
        df_raw = pd.DataFrame(segs)
        st.dataframe(df_raw, use_container_width=True)

# Summarize on demand
if st.button("📝 Generate summary"):
    segs = get_meeting_segments(sel)
    if segs:
        with st.spinner("Clustering & summarizing..."):
            df_sum = summarize_meeting(segs, n_topics, top_n)
        st.subheader("Extractive Summary by Topic")
        st.dataframe(df_sum, use_container_width=True)



