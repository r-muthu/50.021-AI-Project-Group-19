import streamlit as st
import torch
import torch.nn as nn
from transformers import RobertaTokenizerFast, RobertaConfig, RobertaForSequenceClassification
from safetensors.torch import load_file

# --- Load Tokenizer ---
tokenizer = RobertaTokenizerFast.from_pretrained(
    './', tokenizer_file="tokenizer.json"
)

# --- Define model config that matches your trained model ---
config = RobertaConfig.from_pretrained("roberta-base", num_labels=2)

# --- Define the custom wrapper ---
class BaseModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pretrained_model = RobertaForSequenceClassification(config)

    def forward(self, **kwargs):
        return self.pretrained_model(**kwargs)

# --- Instantiate model with config only ---
model = BaseModel(config)

# --- Load weights safely ---
state_dict = load_file("model.safetensors")  # Direct path, as before
model.load_state_dict(state_dict, strict=False)
model.eval()

# --- Streamlit UI ---
st.set_page_config(page_title="🛡️ Hate Speech Detector", layout="centered")
st.title("🛡️ Hate Speech Detector")
st.markdown("Enter a sentence below to classify it as **Hateful** or **Non-Hateful**.")

user_input = st.text_area("💬 Input Text", height=150)

if st.button("🧠 Analyze"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=1).item()

        label = "🟥 Hateful" if prediction == 1 else "🟩 Non-Hateful"
        st.markdown(f"### Prediction: {label}")
