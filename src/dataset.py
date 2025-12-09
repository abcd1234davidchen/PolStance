import torch
from torch.utils.data import Dataset
import pandas as pd
import sqlite3


class StanceDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=64, split=None, mask_keywords=None, mask_prob=0.0):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_keywords = mask_keywords if mask_keywords else []
        self.mask_prob = mask_prob

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        # Keyword Masking Strategy
        if self.mask_keywords and self.mask_prob > 0:
            import random
            if random.random() < self.mask_prob:
                for keyword in self.mask_keywords:
                    if keyword in text:
                        text = text.replace(keyword, self.tokenizer.mask_token)
        
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        item["text"] = text
        return item

def create_dataset(db_path):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT url, article, label, title FROM articleTable", conn)
    df = df.dropna(subset=["article", "label", "title"])
    print(f"Initial dataset size: {len(df)}")
    df = df[df["url"].str.contains("cti") == False]
    print(f"cleaned size: {len(df)}")
    
    df = df[df["article"].str.len() > 0]
    df["label"] = df["label"] - 1
    df = df[df["label"].isin([0, 1, 2])]
    print(f"各標籤數量：\n{df['label'].value_counts()}")
    conn.close()
    return df[["article", "label"]]