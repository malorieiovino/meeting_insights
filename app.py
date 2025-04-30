# app.py
import streamlit as st
import pandas as pd
import re
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import os
import pickle

# Download once on first run
nltk.download('punkt')
nltk.download('stopwords')

# —— Text‐cleaning regexes —— 
FILLERS = {
    'um','uh','hmm','yeah','right','so','okay','okay?','you know',
    'like','i mean','er','ah','mm','ahh','mmm','huh','kay','uhh','umm','well'
}
FILLER_RE = re.compile(r'\b(' + '|'.join(map(re.escape, FILLERS)) + r')\b',
                       flags=re.IGNORECASE)
ANNOT_RE = re.compile(r'\[.*?\]|\(.*?\)')

# —— Feature‐based extractive summary (from your notebook) —— 
action_keywords = ['will', 'need to', 'going to', 'must', 'should', 'action', 'task', 'due']

def extract_features(segments):
    texts = [seg['text_clean'] for seg in segments]
    vect  = TfidfVectorizer(stop_words='english')
    tfidf = vect.fit_transform(texts).toarray()
    doc_vec = tfidf.mean(axis=0).reshape(1, -1)
    sims = cosine_similarity(tfidf, doc_vec).flatten()

    features = []
    for i, seg in enumerate(segments):
        f = {
            'idx':         i,
            'length':      len(seg['text_clean'].split()),
            'sim':         sims[i],
            'contains_a':  any(kw in seg['text_clean'].lower() for kw in action_keywords),
            'is_new_spkr': i == 0 or seg['speaker'] != segments[i-1]['speaker']
        }
        features.append(f)
    return features

def generate_extractive_summary(segments, top_n=3):
    feats = extract_features(segments)
    scores = []
    for f in feats:
        sc = (
            0.4 * f['sim'] +
            0.3 * min(f['length']/20, 1.0) +
            0.2 * int(f['contains_a']) +
            0.1 * int(f['is_new_spkr'])
        )
        scores.append((f['idx'], sc))
    ranked = sorted(scores, key=lambda x: x[1], reverse=True)
    top_idxs = [idx for idx,_ in ranked[:top_n]]
    top_idxs.sort()
    return [segments[i] for i in top_idxs]

# —— Dataset & preprocessing —— 
@st.cache_data(show_spinner=False)
def load_ami_dataset():
    # Read our small sample CSV instead of unpickling
    df = pd.read_csv("data/sample_meeting.csv")
    # Convert to list of dicts for downstream code
    return df.to_dict(orient="records")
	
def clean_segment_text(text):
    text = ANNOT_RE.sub("", text)
    text = FILLER_RE.sub("", text)
    return text.strip()

@st.cache_data
def get_meeting_segments(meeting_id):
    train = load_ami_dataset()
    raw = [item for item in train if item['meeting_id']==meeting_id]
    segments = []
    for seg in raw:
        segments.append({
            'speaker':    seg.get('speaker', 'UNK'),
            'begin_time': seg.get('begin_time', 0.0),
            'end_time':   seg.get('end_time', 0.0),
            'text_clean': clean_segment_text(seg['text'])
        })
    return segments

@st.cache_data
def cluster_into_topics(segments, n_topics):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    texts = [seg['text_clean'] for seg in segments]
    embeddings = model.encode(texts, convert_to_tensor=True).cpu().numpy()
    km = KMeans(n_clusters=n_topics, random_state=42)
    labels = km.fit_predict(embeddings)
    topics = []
    for t in range(n_topics):
        topics.append([seg for seg,lab in zip(segments, labels) if lab==t])
    return topics

def summarize_meeting(segments, n_topics, top_n):
    topics = cluster_into_topics(segments, n_topics)
    rows = []
    for t_idx, topic in enumerate(topics):
        summary = generate_extractive_summary(topic, top_n)
        for seg in summary:
            rows.append({
                'Topic':     t_idx,
                'Speaker':   seg['speaker'],
                'Utterance': seg['text_clean']
            })
    return pd.DataFrame(rows)

# —— Streamlit UI —— 
st.set_page_config(page_title="AMI Meeting Summarizer", layout="wide")
st.title("AMI IHM Meeting Summarization Demo")

# Sidebar controls
st.sidebar.header("⚙️ Parameters")
meeting_ids = sorted({item['meeting_id'] for item in load_ami_dataset()})
sel_meeting = st.sidebar.selectbox("Select meeting_id", meeting_ids)
n_topics    = st.sidebar.slider("Number of topics (clusters)", 2, 10, 5)
top_n       = st.sidebar.slider("Utterances per topic", 1, 5, 3)

# Load & show raw segments
with st.expander("▶️ View raw segments"):
    segs = get_meeting_segments(sel_meeting)
    df_raw = pd.DataFrame(segs)
    st.experimental_data_editor(df_raw, num_rows="dynamic", key="raw")

# Summarize button
if st.button("📝 Generate summary"):
    with st.spinner("Clustering & summarizing..."):
        df_summary = summarize_meeting(segs, n_topics, top_n)
    st.subheader("Extractive Summary by Topic")
    st.dataframe(df_summary, use_container_width=True)


