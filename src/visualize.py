import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import StanceDataset, create_dataset
from transformers import AutoModel , BertTokenizerFast
from model import StanceClassifier

def extract_embeddings(model, loader, device):
    model.eval()
    all_embeddings = []
    all_labels = []
    all_texts = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting Embeddings"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].cpu().numpy()
                
            _, embeddings = model(input_ids=input_ids, attention_mask=attention_mask, return_embeddings=True)

            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.cpu().numpy()
            
            all_embeddings.append(embeddings)
            all_labels.append(labels)
            all_texts.extend(batch['text'])
        
        all_embeddings = np.concatenate(all_embeddings, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
    return all_embeddings, all_labels, all_texts

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

checkpoint = "ckiplab/bert-base-chinese"
tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
base_model = AutoModel.from_pretrained(checkpoint)

model = StanceClassifier(base_model, num_classes=3)
model.load_state_dict(torch.load("stance_classifier.pth", map_location=torch.device('cpu')))
model.to(device)

MAX_LENGTH = 512
BATCH_SIZE = 32

df = create_dataset("article.db")
dataset = StanceDataset(
    texts=df["article"].tolist(),
    labels=df["label"].tolist(),
    tokenizer=tokenizer,
    max_length=512,
)

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# Run the extraction
embeddings, labels, texts = extract_embeddings(model, loader, device=device)

print(f"Extraction complete. Shape: {embeddings.shape}")

df_embeddings = pd.DataFrame({
    "embedding": list(embeddings),
    "label": labels,
    "text": texts,
})
df_embeddings.to_parquet("embeddings.parquet")