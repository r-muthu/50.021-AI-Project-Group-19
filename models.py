from torch import nn
from transformers import AutoModelForSequenceClassification, AutoModel

class BaseModel(nn.Module):
    def __init__(self, model_checkpoint, num_labels=2, hidden_dropout_prob=0.1):
        super().__init__()
        self.pretrained_model = AutoModelForSequenceClassification.from_pretrained(
            model_checkpoint,
            num_labels=num_labels,
            hidden_dropout_prob=hidden_dropout_prob
        )

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        return self.pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
            return_dict=True
        )

    def resize_token_embeddings(self, token_len):
        self.pretrained_model.resize_token_embeddings(token_len)

class CustomClassifier(nn.Module):
    def __init__(self, model_checkpoint, num_labels=2, hidden_dropout_prob=0.1):
        super().__init__()
        self.pretrained_model = AutoModel.from_pretrained(model_checkpoint)
        hidden_size = self.pretrained_model.config.hidden_size
        
        # Freeze all base model parameters
        for param in self.pretrained_model.parameters():
            param.requires_grad = True
        
        # New trainable classification head
        self.final_classifier = nn.Sequential(
            nn.Dropout(hidden_dropout_prob),
            nn.Linear(hidden_size, num_labels)  # Binary classification (0 or 1)
        )

    def resize_token_embeddings(self, token_len):
        self.pretrained_model.resize_token_embeddings(token_len)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        pooled_output = outputs.pooler_output  # [CLS] representation

        logits = self.final_classifier(pooled_output)  # no squeeze
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)  # labels should be LongTensor, e.g., 0 or 1

        return {"loss": loss, "logits": logits}