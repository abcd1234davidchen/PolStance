import torch
import torch.nn as nn
from transformers import AutoModel , BertTokenizerFast
from model import StanceClassifier
    
torch.manual_seed(42)
checkpoint = "ckiplab/bert-base-chinese"
tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
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