import torch
import torch.nn as nn


class StanceClassifier(nn.Module):
    def __init__(self, transformer_model, num_classes, dropout_rate=0.6):
        super(StanceClassifier, self).__init__()
        self.transformer = transformer_model
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(transformer_model.config.hidden_size)
        
        l0 = transformer_model.config.hidden_size
        l1 = transformer_model.config.hidden_size * 2
        l2 = l1 // 2
        l3 = l2 // 2
        # classifier expects pooled token representation (batch, hidden)
        self.classifier = nn.Sequential(
            nn.Linear(l0, l1),
            nn.LayerNorm(l1),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(l1, l2),
            nn.LayerNorm(l2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(l2, l3),
            nn.LayerNorm(l3),
            nn.GELU(),
            nn.Linear(l3, num_classes),
        )

        self.attention_vector = nn.Linear(l0, 1)
        nn.init.xavier_uniform_(self.attention_vector.weight)

        
        self.freeze_transformer()

    def freeze_transformer(self):
        for param in self.transformer.parameters():
            param.requires_grad = False

    def unfreeze_transformer(self):
        for param in self.transformer.parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask,return_embeddings=False):
        if not any(p.requires_grad for p in self.transformer.parameters()):
            with torch.no_grad():
                outputs = self.transformer(
                    input_ids=input_ids, attention_mask=attention_mask
                )
        else:
            outputs = self.transformer(
                input_ids=input_ids, attention_mask=attention_mask
            )

        # token-level hidden states: (batch, seq_len, hidden)
        token_states = outputs.last_hidden_state

        scores = self.attention_vector(token_states).squeeze(-1)  # (batch, seq_len)
        mask = attention_mask.to(dtype=torch.bool)  # (batch, seq_len)
        scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=1)  # (batch, seq_len)
        pooled_output = (weights.unsqueeze(-1) * token_states).sum(dim=1)  # (batch, hidden)

        if torch.isnan(pooled_output).any() or torch.isinf(pooled_output).any():
            print("WARNING: Transformer output NaN/Inf")
            pooled_output = torch.where(
                torch.isnan(pooled_output) | torch.isinf(pooled_output),
                torch.zeros_like(pooled_output),
                pooled_output,
            )

        pooled_output = self.layer_norm(pooled_output)
        logits = self.classifier(pooled_output)
        if return_embeddings:
            return logits, pooled_output
        return logits

    def classifier_params(self):
        return list(self.classifier.parameters())
    
    def transformer_params(self):
        return list(self.transformer.parameters())