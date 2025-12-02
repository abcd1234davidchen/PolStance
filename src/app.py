import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import gradio as gr
import re

class StanceClassifier(nn.Module):
    def __init__(self,transformer_model, num_classes, dropout_rate=0.3):
        super(StanceClassifier, self).__init__()
        self.transformer = transformer_model
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(transformer_model.config.hidden_size)
        l0 = transformer_model.config.hidden_size
        l1 = transformer_model.config.hidden_size * 2
        l2 = l1 // 2
        l3 = l2 // 2
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
        
        
    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.layer_norm(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
    
torch.manual_seed(42)
checkpoint = "hfl/chinese-roberta-wwm-ext"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
base_model = AutoModel.from_pretrained(checkpoint)

model = StanceClassifier(base_model, num_classes=3)
model.load_state_dict(torch.load("stance_classifier.pth", map_location=torch.device('cpu')))
model.eval()
labels = ['KMT', 'DPP', 'Neutral']

def predict_stance(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )
        probs = nn.Softmax()(outputs)
        print(probs)
        predicted_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][predicted_class].item()
    return labels[predicted_class], confidence

def gradio_interface(text):
    sentences = re.split(r"[。！？\n]", text)
    sentences = [s for idx, s in enumerate(sentences) if s.strip()]
    accumulate_sentence = [" ".join(sentences[:idx+1]) for idx, s in enumerate(sentences) if s.strip()]
    results = []
    for s, acus in zip(sentences, accumulate_sentence):
        stance, conf = predict_stance(acus)
        results.append((s + f" (Confidence: {conf:.4f})", stance))
    return results

def ui():
    gr.Interface(
        fn=gradio_interface,
        inputs=gr.Textbox(label="Input Text", placeholder="Enter text to predict political stance..."),
        outputs=gr.HighlightedText(label="Prediction Result",color_map={"KMT":"blue","DPP":"green","Neutral":"purple"}),
        title="Political Stance Prediction",
        description="Enter a text to predict its political stance (KMT, DPP, Neutral)."
    ).launch()

if __name__ == "__main__":
    ui()