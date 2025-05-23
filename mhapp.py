import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Page configuration - first Streamlit command
st.set_page_config(page_title="AI Therapist", layout="centered")

# Load tokenizer and model
@st.cache_resource
def load_model():
    model_name = "bhadresh-savani/distilbert-base-uncased-emotion"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()
id2label = model.config.id2label

# Emotion prediction function
def predict_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = F.softmax(logits, dim=1)
    top_prob, top_label_id = torch.max(probs, dim=1)
    top_label = id2label[top_label_id.item()]
    return top_label, top_prob.item()

# Therapist responses
responses = {
    "joy": "That's wonderful! Tell me more about what's making you feel happy.",
    "sadness": "I'm here for you. Would you like to talk more about what’s making you feel down?",
    "anger": "It's okay to feel angry. Want to share what's upsetting you?",
    "fear": "You're not alone. What's making you feel this way?",
    "surprise": "Oh! That sounds unexpected. Care to explain more?",
    "neutral": "Thanks for sharing. I'm here to listen.",
    "tired": "It's okay to rest. How long have you been feeling this way?",
    "curiosity": "That's great! What are you curious about?",
    "confusion": "I'm here to help. What’s confusing you?",
    # Add more nuanced emotions as needed
}

# UI
st.title("🧠 AI Therapist - How are you feeling today?")
st.markdown("This AI detects your emotion and responds accordingly. Type your thoughts below:")

user_input = st.text_input("Your Message:")

if user_input:
    label, confidence = predict_emotion(user_input)
    st.markdown(f"**Detected Emotion:** {label} ({confidence*100:.2f}% confidence)")

    response = responses.get(label, "Thank you for sharing. I'm listening.")
    st.markdown(f"**Therapist:** {response}")
