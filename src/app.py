import torch
import torch.nn as nn
from transformers import AutoModel, BertTokenizerFast
import gradio as gr
import re
from model import StanceClassifier
import os
import huggingface_hub

torch.manual_seed(42)
checkpoint = "ckiplab/bert-base-chinese"
tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
base_model = AutoModel.from_pretrained(checkpoint)

model = StanceClassifier(base_model, num_classes=3)

dict_path = huggingface_hub.hf_hub_download(repo_id="abcd1234davidchen/PolStanceBERT",filename="stance_classifier.pth",local_dir=".",local_dir_use_symlinks=False)

model.load_state_dict(torch.load(dict_path, map_location=torch.device('cpu')))
model.eval()
labels = ['KMT', 'DPP', 'Neutral']

def predict_stance(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
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
    singleSentenceMode = False
    if text[0:1]=="!" or text[0:1]=="！":
        text=text[1:]
        singleSentenceMode = True

    sentences = re.split(r"[。！？\n]", text)
    sentences = [s for idx, s in enumerate(sentences) if s.strip()]
    accumulate_sentence = [" ".join(sentences[:idx+1]) for idx, s in enumerate(sentences) if s.strip()]
    results = []
    if singleSentenceMode:
        for s in sentences:
            stance, conf = predict_stance(s)
            results.append((s + f" (Confidence: {conf:.4f})", stance))
        return results
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
        description="Enter a text to predict its political stance (KMT, DPP, Neutral). Prefix a sentence with '!' or '！' to analyze each sentence individually.",
    ).launch()

if __name__ == "__main__":
    ui()