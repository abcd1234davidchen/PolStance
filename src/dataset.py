import torch
from torch.utils.data import Dataset
import pandas as pd
import sqlite3


class StanceDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
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
    min_label_count = df["label"].value_counts().min()
    print(f"各標籤數量：\n{df['label'].value_counts()}")
    df = df.groupby("label").sample(n=min_label_count, random_state=42)
    conn.close()
    return df[["article", "label"]]
