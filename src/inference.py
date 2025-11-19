import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class StanceClassifier(nn.Module):
    def __init__(self,transformer_model, num_classes, dropout_rate=0.6):
        super(StanceClassifier, self).__init__()
        self.transformer = transformer_model
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(transformer_model.config.hidden_size)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(transformer_model.config.hidden_size, transformer_model.config.hidden_size//2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(transformer_model.config.hidden_size//2, num_classes)
        )
    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.layer_norm(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
    
torch.manual_seed(42)
checkpoint = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
base_model = AutoModel.from_pretrained(checkpoint)

model = StanceClassifier(base_model, num_classes=3)
model.load_state_dict(torch.load("stance_classifier.pth", map_location=torch.device('cpu')))
model.eval()
labels = ['KMT', 'DPP', 'Neutral']

def predict_stance(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=64)
    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )
        probs = nn.Softmax(dim=1)(outputs)
        print(probs)
        predicted_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][predicted_class].item()
    return labels[predicted_class], confidence

if __name__ == "__main__":
    test_text = "大罷免大成功"
    stance, conf = predict_stance(test_text)
    print(f"Predicted Stance: {stance} with confidence {conf:.4f}")