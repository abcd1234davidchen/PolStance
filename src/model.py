import torch
import torch.nn as nn


class StanceClassifier(nn.Module):
    def __init__(self, transformer_model, num_classes, dropout_rate=0.6):
        super(StanceClassifier, self).__init__()
        self.transformer = transformer_model
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(transformer_model.config.hidden_size)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(
                transformer_model.config.hidden_size,
                transformer_model.config.hidden_size // 2,
            ),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(transformer_model.config.hidden_size // 2, num_classes),
        )
        self.freeze_transformer()

    def freeze_transformer(self):
        for param in self.transformer.parameters():
            param.requires_grad = False

    def unfreeze_transformer(self):
        for param in self.transformer.parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        if not any(p.requires_grad for p in self.transformer.parameters()):
            with torch.no_grad():
                outputs = self.transformer(
                    input_ids=input_ids, attention_mask=attention_mask
                )
        else:
            outputs = self.transformer(
                input_ids=input_ids, attention_mask=attention_mask
            )

        pooled_output = outputs.last_hidden_state[:, 0]

        if torch.isnan(pooled_output).any() or torch.isinf(pooled_output).any():
            print("WARNING: Transformer output NaN/Inf")
            pooled_output = torch.where(
                torch.isnan(pooled_output) | torch.isinf(pooled_output),
                torch.zeros_like(pooled_output),
                pooled_output,
            )

        pooled_output = self.layer_norm(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
