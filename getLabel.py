import base64
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv
import sqlite3
import json
import tqdm

load_dotenv()

def generateStance(articleText):
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    model = "gemini-flash-latest"
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
            required = ["偏向國民黨","偏向民進黨","偏向民眾黨","中立"],
            properties = {
                "偏向國民黨": genai.types.Schema(
                    type = genai.types.Type.INTEGER,
                ),
                "偏向民進黨": genai.types.Schema(
                    type = genai.types.Type.INTEGER,
                ),
                "偏向民眾黨": genai.types.Schema(
                    type = genai.types.Type.INTEGER,
                ),
                "中立": genai.types.Schema(
                    type = genai.types.Type.INTEGER,
                ),
            },
        ),
        system_instruction='''你是一個政治立場分析專家，請根據使用者提供的文章內容，分析該文章的政治立場傾向，並以JSON格式回覆，包含以下欄位：偏向國民黨、偏向民進黨、偏向民眾黨、中立。每個欄位的值應為0或1，文章只能有一個傾向，表示該文章對應政治立場的傾向程度。''',
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
        print(f"Raw response: {full_response}")
        
        labels = json.loads(full_response)
        print(f"Parsed labels: {labels}")
        return labels
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        print(f"Raw response was: {full_response}")
        return None
    except Exception as e:
        print(f"Error generating stance: {e}")
        return None


def readDB():
    conn = sqlite3.connect('articles.db')
    cursor = conn.cursor()
    # cursor.execute("""
    #     SELECT id, url, text FROM articles 
    #     WHERE text IS NOT NULL AND text != '' 
    # """)
    cursor.execute("""
        SELECT id, url, text FROM articles 
        WHERE text IS NOT NULL AND text != '' 
        AND (
            (label_kmt = -1 OR label_dpp = -1 OR label_tpp = -1 OR label_neutral = -1)
            OR 
            (label_kmt = 0 AND label_dpp = 0 AND label_tpp = 0 AND label_neutral = 0)
        )
    """)
    rows = cursor.fetchall()
    conn.close()
    return rows

def updateArticleLabels(article_id, labels):
    conn = sqlite3.connect('articles.db')
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE articles
        SET label_kmt = ?, label_dpp = ?, label_tpp = ?, label_neutral = ?
        WHERE id = ?
    ''', (labels.get("偏向國民黨", 0), labels.get("偏向民進黨", 0), labels.get("偏向民眾黨", 0), labels.get("中立", 0), article_id))
    conn.commit()
    conn.close()

def labelArticles():
    rows = readDB()
    for row in tqdm.tqdm(rows):
        article_id, url, text = row
        print(f"Processing article ID {article_id} from URL {url}")
        try:
            labels = generateStance(text)
            if labels and isinstance(labels, dict):
                updateArticleLabels(article_id, labels)
                print(f"Updated labels for article ID {article_id}: {labels}")
            else:
                print(f"No labels generated for article ID {article_id}")
        except Exception as e:
            print(f"Error processing article ID {article_id}: {e}")

if __name__ == "__main__":
    labelArticles()