import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn.functional as F

st.set_page_config(page_title="AI Therapist", layout="centered")
model_name = "j-hartmann/emotion-english-distilroberta-base"


@st.cache_resource(show_spinner=True)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()
id2label = model.config.id2label

def predict_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = F.softmax(logits, dim=1)
    top_prob, top_label_id = torch.max(probs, dim=1)
    top_label = id2label[top_label_id.item()]
    return top_label, top_prob.item()

st.title("🧠 AI Therapist - How are you feeling today?")
st.write("This AI detects your emotion and responds accordingly. Type your thoughts below:")

user_input = st.text_area("Your Message:", height=150)

if st.button("Submit"):
    if user_input.strip() == "":
        st.warning("Please type something to analyze.")
    else:
        emotion, confidence = predict_emotion(user_input)
        st.markdown(f"**Detected Emotion:** {emotion} ({confidence:.2%} confidence)")
        responses = {
            "admiration": "It's wonderful that you recognize and appreciate greatness!",
            "joy": "That's amazing! It's great to see you so happy. Tell me more about what’s making you feel this way.",
            "sadness": "I'm here for you. Would you like to talk more about what’s making you feel down?",
            "anger": "It’s okay to feel angry. Do you want to share what triggered it?",
            "fear": "That sounds scary. I’m here with you. Would you like to talk more about it?",
            "disgust": "Ugh, I get that. Some things really do feel unpleasant. Want to vent?",
            "neutral": "Thanks for sharing. Would you like to go deeper?"
        }
        response = responses.get(emotion, "Thanks for sharing. I'm here to listen.")
        st.success(f"**Therapist:** {response}")
