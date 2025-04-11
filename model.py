import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import RobertaTokenizer, RobertaForSequenceClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

bert_model = BertForSequenceClassification.from_pretrained("path/to/fine-tuned-bert").to(device)
roberta_model = RobertaForSequenceClassification.from_pretrained("path/to/fine-tuned-roberta").to(device)

bert_model.eval()
roberta_model.eval()

def get_model_output(text, model, tokenizer, return_probs=True):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        if return_probs:
            return F.softmax(logits, dim=-1)
        else:
            return torch.argmax(logits, dim=-1)

def soft_voting(text):
    bert_probs = get_model_output(text, bert_model, bert_tokenizer, return_probs=True)
    roberta_probs = get_model_output(text, roberta_model, roberta_tokenizer, return_probs=True)

    avg_probs = (bert_probs + roberta_probs) / 2
    pred_class = torch.argmax(avg_probs, dim=-1).item()
    return pred_class, avg_probs.squeeze().tolist()

def hard_voting(text):
    bert_pred = get_model_output(text, bert_model, bert_tokenizer, return_probs=False).item()
    roberta_pred = get_model_output(text, roberta_model, roberta_tokenizer, return_probs=False).item()

    votes = [bert_pred, roberta_pred]
    pred_class = max(set(votes), key=votes.count)
    return pred_class, votes

text = "You're such an idiot!"

pred_soft, soft_probs = soft_voting(text)
print(f"[Soft Voting] Predicted class: {pred_soft}, Probabilities: {soft_probs}")

pred_hard, votes = hard_voting(text)
print(f"[Hard Voting] Predicted class: {pred_hard}, Votes: {votes}")
