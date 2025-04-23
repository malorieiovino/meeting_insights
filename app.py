# app.py
import streamlit as st
from datasets import load_dataset
import numpy as np
import pandas as pd

# — import your pipeline functions —
from meeting_insights import (
    segment_topics_improved,
    generate_extractive_summary,
    extract_action_items_enhanced,
    plot_confidence_heatmap,
    plot_priority_distribution
)

st.set_page_config(layout="wide")
st.title("💡 Meeting Insights")

@st.cache(allow_output_mutation=True)
def load_demo_meeting():
    ds = load_dataset("edinburghchr/ami", "ihm", split="train")
    # pick first meeting as demo
    mid = ds[0]["meeting_id"]
    segs = [
        {"text": rec["text"], "speaker": rec["speaker_id"]}
        for rec in ds if rec["meeting_id"] == mid
    ]
    return mid, segs

meeting_id, segments = load_demo_meeting()
st.sidebar.markdown(f"**Demo meeting:** {meeting_id}  \nSegments: {len(segments)}")

if st.sidebar.button("Run full pipeline"):
    # 1) topic segmentation
    topics = segment_topics_improved(segments)
    st.header("1. Topic Segmentation")
    lengths = [len(t) for t in topics]
    st.write(f"Detected {len(topics)} topics; utterances per topic:", lengths)

    # 2) extractive summarization
    st.header("2. Summaries (top 3 per topic)")
    all_summ = []
    for i, top in enumerate(topics, start=1):
        segs = generate_extractive_summary(top, top_n=3)
        for seg in segs:
            all_summ.append({
                "Topic": i,
                "Speaker": seg["speaker"],
                "Summary": seg.get("text_clean", seg["text"])
            })
    df_summ = pd.DataFrame(all_summ)
    st.dataframe(df_summ, height=300)

    # 3) action‐item extraction
    st.header("3. Action Items")
    all_actions = []
    for i, top in enumerate(topics, start=1):
        acts = extract_action_items_enhanced(top)
        for rank, a in enumerate(acts, start=1):
            dl = (a.get("deadline") or {}).get("term", "")
            all_actions.append({
                "Topic": i,
                "Rank": rank,
                "Owner": a["owner"],
                "Task": a["task"],
                "Priority": a["priority"],
                "Deadline": dl,
                "Confidence": a["confidence"]
            })
    df_act = pd.DataFrame(all_actions)
    st.dataframe(df_act, height=300)

    # 4) Visualizations
    st.header("4. Visualizations")
    st.subheader("Confidence Heatmap")
    fig1 = plot_confidence_heatmap(topics, return_fig=True)
    st.pyplot(fig1)

    st.subheader("Priority Distribution")
    fig2 = plot_priority_distribution(topics, return_fig=True)
    st.pyplot(fig2)

