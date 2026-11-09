import streamlit as st
import torch
import re
import time

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification
)

st.set_page_config(
    page_title="CampusIntent AI",
    page_icon="🎓",
    layout="centered"
)

MODEL_PATH = "models/distilbert"

# Mask personal information
def mask_personal_data(text):
    text = re.sub(
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b',
        '[EMAIL]',
        text
    )

    text = re.sub(
        r'\b\d{10}\b',
        '[PHONE]',
        text
    )

    return text


# Load model
@st.cache_resource
def load_model():
    tokenizer = DistilBertTokenizerFast.from_pretrained(
        MODEL_PATH
    )

    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_PATH
    )

    model.eval()

    return tokenizer, model


tokenizer, model = load_model()

# Application UI
st.title("🎓 CampusIntent AI")
st.subheader("Student Query Intent Classification")

st.write(
    "Enter a campus-related student query. "
    "The AI model will identify its intent."
)

query = st.text_area(
    "Enter your query",
    placeholder="Example: How can I pay my semester fees?"
)

if st.button("Classify Query", type="primary"):

    if query.strip() == "":
        st.warning("Please enter a query.")

    else:
        masked_query = mask_personal_data(query)

        start_time = time.time()

        inputs = tokenizer(
            masked_query,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=64
        )

        with torch.no_grad():
            outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )

        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )

        predicted_intent = model.config.id2label[
            prediction.item()
        ]

        latency = (time.time() - start_time) * 1000

        st.success(
            f"Predicted Intent: {predicted_intent}"
        )

        st.metric(
            "Confidence",
            f"{confidence.item() * 100:.2f}%"
        )

        st.metric(
            "Inference Latency",
            f"{latency:.2f} ms"
        )

        st.write("### Privacy Protection")

        if masked_query != query:
            st.info(
                f"Masked query: {masked_query}"
            )
        else:
            st.info(
                "No email or 10-digit phone number detected."
            )

        st.write("### Intent Probabilities")

        for i, probability in enumerate(
            probabilities[0].tolist()
        ):
            label = model.config.id2label[i]

            st.write(
                f"{label}: {probability * 100:.2f}%"
            )

            st.progress(probability)

st.divider()

st.caption(
    "CampusIntent AI | DistilBERT | "
    "Generative AI Mini Project"
)
