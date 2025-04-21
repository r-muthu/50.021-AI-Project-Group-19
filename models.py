import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from transformers import AutoModelForSequenceClassification, AutoModel, AutoConfig

class BaseModel(nn.Module):
    def __init__(self, model_checkpoint, num_labels=2, hidden_dropout_prob=0.1):
        super().__init__()
        self.config = AutoConfig.from_pretrained(
            model_checkpoint,
            num_labels=num_labels,
            hidden_dropout_prob=hidden_dropout_prob
        )
        
        self.pretrained_model = AutoModelForSequenceClassification.from_pretrained(
            model_checkpoint,
            config=self.config
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

    
class SimpleTransformerModel(nn.Module):
    def __init__(self, vocab_size, num_labels=2, embed_dim=256, ff_dim=512, num_heads=4, num_layers=3, max_seq_length=512, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim

        # Embedding layers
        self.token_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.position_embeddings = nn.Embedding(max_seq_length, embed_dim)

        # Transformer encoder stack
        encoder_layer = TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers)

        # BatchNorm + Dropout after each layer
        self.norm_layers = nn.ModuleList([nn.BatchNorm1d(embed_dim) for _ in range(num_layers)])
        self.dropout_layers = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_layers)])

        # Classifier head
        self.classifier = nn.Linear(embed_dim, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None):
        batch_size, seq_length = input_ids.size()
        device = input_ids.device

        # Create position ids
        position_ids = torch.arange(0, seq_length, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)

        # Embed tokens + positions
        x = self.token_embeddings(input_ids) + self.position_embeddings(position_ids)

        # Transformer + BatchNorm + Dropout
        for i, layer in enumerate(self.transformer_encoder.layers):
            x = layer(x)
            x = self.norm_layers[i](x.transpose(1, 2)).transpose(1, 2)  # BatchNorm1d expects (B, C, L)
            x = self.dropout_layers[i](x)

        # Mean pooling across sequence length
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).expand_as(x).float()
            x = (x * mask).sum(1) / mask.sum(1)
        else:
            x = x.mean(dim=1)

        logits = self.classifier(x)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return {"loss": loss, "logits": logits}
    
