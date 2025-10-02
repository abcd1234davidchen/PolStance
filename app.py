import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import gradio as gr

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

def gradio_interface(text):
    stance, conf = predict_stance(text)
    return f"Predicted Stance: {stance} with confidence {conf:.4f}"

def ui():
    gr.Interface(
        fn=gradio_interface,
        inputs=gr.Textbox(label="Input Text", placeholder="Enter text to predict political stance..."),
        outputs=gr.Textbox(label="Prediction Result"),
        title="Political Stance Prediction",
        description="Enter a text to predict its political stance (KMT, DPP, Neutral)."
    ).launch()

if __name__ == "__main__":
    ui()