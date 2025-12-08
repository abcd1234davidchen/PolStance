import torch
from torch.utils.data import Dataset
import pandas as pd
import sqlite3


class StanceDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=64, split=None):
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

# drop "【更多內容 請見影片】訂閱【自 由追新聞】全新的視界！新聞話題不漏接，快訂閱YouTube 【自由追新聞】，記得開啟小鈴鐺哦！"
# drop "訂閱【自由追新聞】全新的視界！新聞 話題不漏接，快訂閱YouTube 【自由追新聞】，記得開啟小鈴鐺哦！"
'''deleted """
// 創物件 var tvPlayer = new VideoAPI_LiTV(); // 設定自動播放 tvPlayer.setAutoplay(true); //不自動播放 tvPlayer.setDelay(0); // 設定延遲 tvPlayer.setAllowFullscreen(true); tvPlayer.setType('web'); // tvPlayer.setControls(1); litv 無法操作顯示控制項 tvPlayer.pushVideoIdByClassName('TVPlayer', tvPlayer); setTimeout(function (){ tvPlayer.loadAPIScript('cache_video_js_LiTV'); },3000)
"""
'''