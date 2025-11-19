import base64
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv
import sqlite3
import json
import tqdm
import pandas as pd

load_dotenv()
db_istructure = lambda row: (row[0], row[1], row[2]) # id, url, title

def generateStance(articleText):
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    model = "gemini-flash-lite-latest"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=articleText),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        thinking_config = types.ThinkingConfig(
            thinking_budget=0,
        ),
        response_mime_type="application/json",
        response_schema=genai.types.Schema(
            type = genai.types.Type.OBJECT,
            required = ["a", "b", "c", "d", "e","f","g","h","i","j","k","l"],
            properties = {
                "a": genai.types.Schema(
                    type = genai.types.Type.INTEGER,
                ),
                "b": genai.types.Schema(
                    type = genai.types.Type.INTEGER,
                ),
                "c": genai.types.Schema(
                    type = genai.types.Type.INTEGER,
                ),
                "d": genai.types.Schema(
                    type = genai.types.Type.INTEGER,
                ),
                "e": genai.types.Schema(
                    type = genai.types.Type.INTEGER,
                ),
                "f": genai.types.Schema(
                    type = genai.types.Type.INTEGER,
                ),
                "g": genai.types.Schema(
                    type = genai.types.Type.INTEGER,
                ),
                "h": genai.types.Schema(
                    type = genai.types.Type.INTEGER,
                ),
                "i": genai.types.Schema(
                    type = genai.types.Type.INTEGER,
                ),
                "j": genai.types.Schema(
                    type = genai.types.Type.INTEGER,
                ),
                "k": genai.types.Schema(
                    type = genai.types.Type.INTEGER,
                ),
                "l": genai.types.Schema(
                    type = genai.types.Type.INTEGER,
                ),
            },
        ),
        system_instruction=[
            types.Part.from_text(text="""你是一個政治立場分析專家，請根據使用者提供的多個標題，分析每個標題的政治立場傾向，並以JSON格式回覆
若傾向國民黨則標示為\"1\"，若傾向民進黨則標示為\"2\"，若為中立則標示為\"3\"
文章只能有一個傾向。"""),
        ],
    )
    try:
        chunks = []
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            if hasattr(chunk, 'text'):
                chunks.append(chunk.text)
        
        full_response = ''.join(chunks)
        # print(f"Raw response: {full_response}")
        
        labels = json.loads(full_response)
        # print(f"Parsed labels: {labels}")
        return labels
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        print(f"Raw response was: {full_response}")
        return None
    except Exception as e:
        print(f"Error generating stance: {e}")
        return None


def readDB():
    conn = sqlite3.connect('title.db')
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, url, title FROM titles 
        WHERE title IS NOT NULL AND title != '' 
        AND (label IS NULL OR label NOT IN (1,2,3))
    """)
    rows = cursor.fetchall()
    conn.close()
    return rows

def updateArticleLabels(row_datas, labels):
    conn = sqlite3.connect('title.db')
    cursor = conn.cursor()
    
    
    for idx, (article_id, _, texts) in enumerate(row_datas.keys()):
        cursor.execute('''
            UPDATE titles
            SET label = ?
            WHERE id = ?
        ''', (labels.get(f"{chr(ord('a')+idx)}", 0), article_id))
    conn.commit()
    conn.close()

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

        combined_text = (f"{(chr(ord('a')+idx))}. {article_id}{"\n" if idx != 11 else ""}" for idx, (article_id, _, text) in enumerate(row_data.values()))
        # "a. title1\n b. title2\n c. title3\n ..."
        
        batch_num = (i // 12) + 1
        pbar.set_description(f"Labeling Batch {batch_num} ID: {row_data[0][0]}~{row_data[11][0]}")
        try:
            raise Exception("Testing error handling")  # Remove or comment this line in production
            labels = generateStance(combined_text)
            if labels and isinstance(labels, dict):
                updateArticleLabels(row_data, labels)
            else:
                tqdm.tqdm.write(f"No labels generated for article ID {article_ida}")
        except Exception as e:
            print(f"Error processing article: {e}")

if __name__ == "__main__":
    labelArticles()
    labelArticles()
    conn = sqlite3.connect('title.db')
    df = pd.read_sql_query("SELECT title, label FROM titles", conn)
    df = df.dropna(subset=['title', 'label'])
    df = df[df['title'].str.len() > 0]
    df['label'] = df['label']-1
    df = df[df['label'].isin([0, 1, 2])]
    min_label_count = df['label'].value_counts().min()
    print(f"各標籤數量：\n{df['label'].value_counts()}")