<h2 align="center">CampusIntent AI</h2>

<p align="center"><sub>Student query intent classification: prompting vs fine-tuning<br>Generative AI mini-project.</sub></p>

CampusIntent AI is a Streamlit app that reads a campus-related student query and predicts its intent. It also shows the model's confidence, inference latency, a privacy check, and the probability of every intent.

### Intents


Admissions, Exams, Fees, Hostel, Placements, Scholarships

### Features

- Intent prediction for free-text student queries
- Confidence score and inference latency for every prediction
- Per-intent probability bars
- Privacy check that flags emails and 10-digit phone numbers in a query
- Comparison of three approaches on a 100-query labelled test set:
  1. Few-shot prompted LLM
  2. LoRA-tuned small LLM
  3. Fine-tuned DistilBERT (TensorFlow 2)
- Rare intents augmented with LLM-generated training queries

### Screenshots

![Prediction](assets/screenshot-1.png)
![Probabilities and privacy check](assets/screenshot-2.png)

### Tech Stack

- Python
- Streamlit
- TensorFlow 2
- Hugging Face Transformers (DistilBERT)

### Project Structure

```
CampusIntent-AI/
├── app.py              # Streamlit app
├── models/
│   └── distilbert/     # Fine-tuned model (weights not in the repo, see below)
├── assets/             # Screenshots
├── requirements.txt
└── README.md
```

### Setup

```
git clone https://github.com/khushboo-yadav04/CampusIntent-AI.git
cd CampusIntent-AI
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Model Weights

The weights file `models/distilbert/model.safetensors` (about 255 MB) is larger than GitHub's 100 MB file limit, so it is not stored in this repository.

To get the model, either train it yourself or download it from the link below and place it at `models/distilbert/model.safetensors`.

Download link: _add your Hugging Face or Google Drive link here_

### Run

```
streamlit run app.py
```

Open http://localhost:8501 in your browser, type a query and click **Classify Query**.

### Example

Query: `When will my scholarship amount be credited?`

The app predicts **Fees** with 18.72% confidence, while Scholarships scores 17.78%. This is a known failure case: the probabilities are close together, so low-confidence results should be treated as suggestions and not final answers.

### Results

| Model | Accuracy | Latency | Cost |
|---|---|---|---|
| Few-shot LLM | add | add | add |
| LoRA-tuned small LLM | add | add | add |
| DistilBERT (TF2) | add | add | add |

### Limitations

- Predictions are low-confidence on some queries, so wrong labels can happen.
- The first request is slower because the model loads on first use.
- The test set has only 100 queries and mostly standard English.

### Privacy

Queries are checked for emails and 10-digit phone numbers before classification. Do not enter real personal data.

### Author

Khushboo, B.Tech CSE-AI, MIT-ADT University, Pune
GitHub: [khushboo-yadav04](https://github.com/khushboo-yadav04)
