# Meeting Insights

A Python toolkit and demo app for automatically processing meeting transcripts to:

1. **Extract concise, extractive summaries** of key discussion points  
2. **Identify and categorize action items** with assigned owners, priorities, and deadlines  
3. **Segment conversations by topic** for easier navigation and per-topic summaries  

This repo contains both the core processing library (`meeting_insights.py`) and a simple demo UI (e.g. Streamlit/Gradio) that you can deploy on GitHub Pages, Heroku, or Hugging Face Spaces.

---

## 🚀 Features

- **Data loading & preprocessing**: Compatible with the [AMI meeting corpus](https://huggingface.co/datasets/edinburghcstr/ami) (IHM transcripts), but can be extended to any turn-by-turn transcript.  
- **Topic segmentation**: Sliding-window TF-IDF cosine drops with minimum-segment-size smoothing.  
- **Extractive summarization**: TF-IDF / position / speaker novelty-based scoring.  
- **Action-item extraction**: Regex + lightweight POS filters → owner resolution → priority/deadline estimation → confidence scoring.  
- **Interactive demo**: View topic word-clouds, summaries, action-items tables.  

---

## 📦 Installation

```bash
# clone this repo
git clone https://github.com/<your-username>/meeting_insights.git
cd meeting_insights

# create & activate a venv (optional but recommended)
python3 -m venv .venv && source .venv/bin/activate

# install runtime dependencies
pip install -r requirements.txt
