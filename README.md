# Meeting Insights Pipeline

This repo holds a Python script (`meeting_insights.py`) that runs:
1. Topic segmentation on AMI Corpus transcripts  
2. Extractive summarization per segment  
3. Action-item extraction with owner/priority/deadline  

## Usage

```bash
pip install -r requirements.txt
python3 meeting_insights.py --input path/to/transcript.json

