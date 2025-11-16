
import os
import sqlite3
#from huggingface_hub import HfApi
from dotenv import load_dotenv
import tqdm

from geminiLabeling import GeminiLabeling
from gptLabeling import GptLabeling
from deepseekLabeling import DeepSeekLabeling

load_dotenv()
# token = os.getenv("DATASET_KEY")
# hfapi = HfApi(token=token)
# table_name = "articles"

# origin_dataset = hfapi.dataset_info("TWCKaijin/PolStance")
# print(origin_dataset.id)

geminiClient = GeminiLabeling()
gptClient = GptLabeling()
dsClient = DeepSeekLabeling()

def readDB():
    conn = sqlite3.connect('titles.db')
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, url, title FROM titles 
        WHERE title IS NOT NULL AND title != '' 
    """)
    rows = cursor.fetchall()
    conn.close()
    return rows

def labelArticles():
    rows = readDB()
    pbar = tqdm.tqdm(range(0, len(rows), 12), total=(len(rows)+11)//12, desc="Labeling Articles")
    for i in pbar:
        # 取出12篇，若不足則補最後一篇
        def get_row(idx):
            return rows[idx] if idx < len(rows) else rows[i]
        
        row_data = {}
        for offset in range(0, 12):
            article_id, url, title = get_row(i+offset)
            row_data[offset] = (article_id, url, title)

        combined_text = "\n".join([f"{(chr(ord('a')+idx))}. {title}" for idx, (article_id, _, title) in enumerate(row_data.values())])
        # "a. title1\n b. title2\n c. title3\n ..."
        print("Combined Text for labeling:\n"+combined_text)
        batch_num = (i // 12) + 1
        pbar.set_description(f"Labeling Batch {batch_num} ID: {row_data[0][0]}~{row_data[11][0]}")
        try:
            labels_gemini = geminiClient.labeling(combined_text)
            labels_gpt = gptClient.labeling(combined_text)
            labels_ds = dsClient.labeling(combined_text)
            
            print(f"Gemini Labels: {labels_gemini}")
            print(f"GPT Labels: {labels_gpt}")
            print(f"DeepSeek Labels: {labels_ds}")
        except Exception as e:
            print(f"Error processing article: {e}")
if __name__ == "__main__":
    labelArticles()
