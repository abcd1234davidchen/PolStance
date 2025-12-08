import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import StanceDataset, create_dataset
from google import genai
from google.genai import types
import os
import time

def extract_embeddings(loader):
    all_embeddings = []
    all_texts = []
    client1 = genai.Client(api_key="AIzaSyDbMbHYaVk0FVLOaihxsrJOhR_peuKRo1w")
    client2 = genai.Client(api_key="AIzaSyAL4_99Fvkz8VTk_oyavuye-aucTBtyVYA")
    instruction = "分析這個句子的立場是國民黨、民進黨或中立言論"
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Extracting Embeddings")):
            format_text = [f"{instruction}\n{text}" for text in batch]

            if batch_idx % 2 == 0:
                result = client1.models.embed_content(
                    model="gemini-embedding-001",
                    contents=format_text,
                    config=types.EmbedContentConfig(output_dimensionality=768)
                )
            else:
                result = client2.models.embed_content(
                    model="gemini-embedding-001",
                    contents=format_text,
                    config=types.EmbedContentConfig(output_dimensionality=768)
                )
                time.sleep(2)

            for embedding in result.embeddings:
                all_embeddings.append(embedding.values)
            all_texts.extend(batch)
        
    return all_embeddings, all_texts


MAX_LENGTH = 512
BATCH_SIZE = 32

df = create_dataset("article.db")

loader = DataLoader(df['article'].tolist(), batch_size=BATCH_SIZE, shuffle=False)

# Run the extraction
embeddings, texts = extract_embeddings(loader)


df_embeddings = pd.DataFrame({
    "embedding": list(embeddings),
    "label": df["label"].tolist(),
    "text": texts
})
df_embeddings.to_parquet("embeddings.parquet")