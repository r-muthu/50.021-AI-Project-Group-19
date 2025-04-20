import torch
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from data_preprocessing import CustomDataset
from models import BaseModel
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

# --- Config ---
MODEL_CHECKPOINT = "roberta-base"
DATA_PATH = "HateSpeechDatasetBalanced.csv"
SAVE_DIR = "./saved_roberta"
NUM_LABELS = 2
EPOCHS = 3
BATCH_SIZE = 128
MAX_LEN = 128

# --- CUDA Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Using device: {device}")

# --- Load Dataset ---
dataset = CustomDataset(DATA_PATH, model_checkpoint=MODEL_CHECKPOINT, max_len=MAX_LEN)
train_dataset, val_dataset, test_dataset = dataset.get_splits()
tokenizer = dataset.get_tokenizer()

# --- Load Model ---
model = BaseModel(model_checkpoint=MODEL_CHECKPOINT, num_labels=NUM_LABELS)

for param in model.pretrained_model.roberta.encoder.layer[:10].parameters():
    param.requires_grad = False


model.to(device)

# --- Training Args ---
training_args = TrainingArguments(
    output_dir=SAVE_DIR,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-4,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    logging_dir=f"{SAVE_DIR}/logs",
    load_best_model_at_end=True,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),  # use mixed precision if on CUDA
)

# --- Define Trainer ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

print("🚀 Training started...")
trainer.train()

# --- Save final model and tokenizer ---
trainer.save_model(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print(f"✅ Model and tokenizer saved to {SAVE_DIR}")

def test_and_visualize(trainer, test_dataset, label_names=None):
    print("🔍 Running evaluation on test set...")

    # Get predictions
    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)
    y_true = predictions.label_ids

    # Label names
    if label_names is None:
        label_names = ["Not Hate Speech", "Hate Speech"]

    # --- Classification Report ---
    print("\n📋 Classification Report:\n")
    print(classification_report(y_true, y_pred, target_names=label_names))

    # --- Confusion Matrix ---
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    disp.plot(cmap="Blues", values_format='d')
    plt.title("📊 Confusion Matrix")
    plt.show()

    # --- ROC Curve ---
    if predictions.predictions.shape[1] == 2:  # binary classification
        y_scores = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=1)[:, 1].numpy()
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("🧪 ROC Curve")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()

    print("✅ Evaluation complete.")

tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT, use_fast=True)
tokenizer.save_pretrained(SAVE_DIR)


loaded_tokenizer = AutoTokenizer.from_pretrained(SAVE_DIR)
loaded_model = AutoModelForSequenceClassification.from_pretrained(SAVE_DIR)

# Send model to the correct device
loaded_model.to(device)

# Recreate Trainer for evaluation
eval_trainer = Trainer(
    model=loaded_model,
    args=training_args,
    eval_dataset=test_dataset,
    tokenizer=loaded_tokenizer,
)

# Run the test and visualization
test_and_visualize(eval_trainer, test_dataset)

