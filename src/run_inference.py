import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import StanceDataset, create_dataset
from transformers import AutoModel, BertTokenizerFast
from model import StanceClassifier
from dotenv import load_dotenv

def inference(model, loader, device):
    model.eval()
    all_embeddings = []
    all_true_labels = []
    all_pred_labels = []
    all_texts = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Running Inference"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].cpu().numpy()
            
            # Forward pass: get both logits and embeddings
            logits, embeddings = model(input_ids=input_ids, attention_mask=attention_mask, return_embeddings=True)

            # Get predictions
            predictions = torch.argmax(logits, dim=1).cpu().numpy()
            
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.cpu().numpy()
            
            all_embeddings.append(embeddings)
            all_true_labels.append(labels)
            all_pred_labels.append(predictions)
            all_texts.extend(batch['text'])
        
        all_embeddings = np.concatenate(all_embeddings, axis=0)
        all_true_labels = np.concatenate(all_true_labels, axis=0)
        all_pred_labels = np.concatenate(all_pred_labels, axis=0)
        
    return all_embeddings, all_true_labels, all_pred_labels, all_texts

def main():
    load_dotenv()
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Model
    # We use the Stage 2 model ("stance_classifier.pth") which has the trained classifier head
    # AND the (frozen) trained encoder from Stage 1.
    CHECKPOINT_PATH = "stance_classifier.pth"
    print(f"Loading model from {CHECKPOINT_PATH}...")
    
    checkpoint = "ckiplab/bert-base-chinese"
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
    base_model = AutoModel.from_pretrained(checkpoint)
    base_model.resize_token_embeddings(len(tokenizer))
    
    model = StanceClassifier(base_model, num_classes=3)
    
    # Load weights
    # Note: StanceClassifier structure must match what was saved.
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.to(device)

    # Data Preparation
    BATCH_SIZE = 32
    MAX_LENGTH = 512
    
    # Load all data for inference
    print("Loading dataset...")
    df = create_dataset("article.db")
    print(f"Total samples: {len(df)}")

    dataset = StanceDataset(
        texts=df["article"].tolist(),
        labels=df["label"].tolist(),
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
    )

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Run Inference
    embeddings, true_labels, pred_labels, texts = inference(model, loader, device=device)

    print(f"Inference complete. Embeddings shape: {embeddings.shape}")

    # Create DataFrame
    df_results = pd.DataFrame({
        "embedding": list(embeddings),
        "true_label": true_labels,
        "predicted_label": pred_labels,
        "text": texts
    })

    # Save to Parquet
    OUTPUT_FILE = "embeddings_with_predictions.parquet"
    df_results.to_parquet(OUTPUT_FILE)
    print(f"Results saved to {OUTPUT_FILE}")
    print(df_results.head())

if __name__ == "__main__":
    main()
