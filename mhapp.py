import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn.functional as F

st.set_page_config(page_title="AI Therapist", layout="centered")
model_name = "bhadresh-savani/distilbert-base-uncased-emotion"


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
                "amusement": "Haha, that sounds fun! What made you laugh?",
                "anger": "It’s okay to feel angry. Do you want to share what triggered it?",
                "annoyance": "That must have been frustrating. I'm here if you want to talk about it.",
                "approval": "It’s great that you support this! What do you like about it?",
                "caring": "That’s so kind of you. Who or what do you care about?",
                "confusion": "I hear you. Want help sorting out your thoughts?",
                "curiosity": "That’s a great thing to feel. What are you curious about?",
                "desire": "Wishing for something? What do you desire?",
                "disappointment": "That sounds disheartening. I'm here if you want to talk about it.",
                "disapproval": "It’s okay not to agree. What concerns you?",
                "disgust": "Ugh, I get that. Some things really do feel unpleasant. Want to vent?",
                "embarrassment": "That must have been tough. Want to tell me what happened?",
                "excitement": "That’s thrilling! I’d love to hear what has you so excited.",
                "fear": "That sounds scary. I’m here with you. Would you like to talk more about it?",
                "gratitude": "That’s lovely. What are you thankful for?",
                "grief": "I'm so sorry. That must be really hard. I'm here for you.",
                "joy": "That's amazing! Tell me more about what’s making you feel this way.",
                "love": "That’s beautiful. Love is such a powerful feeling. Care to share more?",
                "nervousness": "It’s okay to be nervous. What's making you feel this way?",
                "neutral": "Thanks for sharing. Would you like to go deeper?",
                "optimism": "That’s a great outlook! What’s making you feel hopeful?",
                "pride": "You should be proud! Tell me about your achievement.",
                "realization": "That’s a powerful insight. What led to it?",
                "relief": "I'm glad you're feeling better. What changed?",
                "remorse": "It’s okay to feel regret. Talking about it may help.",
                "sadness": "I'm here for you. Want to talk more about it?",
                "surprise": "Oh wow! What happened?",
                "tiredness": "It’s okay to rest. Do you want to share what’s draining you?",
                # fallback
                "default": "Thanks for sharing. I'm here to listen."
        }
        response = responses.get(emotion, "Thanks for sharing. I'm here to listen.")
        st.success(f"**Therapist:** {response}")
