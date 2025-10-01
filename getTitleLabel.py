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

def updateArticleLabels(article_ida, article_idb, article_idc, article_idd,
                         article_ide, article_idf, article_idg, article_idh,
                         article_idi, article_idj, article_idk, article_idl, labels):
    conn = sqlite3.connect('title.db')
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE titles
        SET label = ?
        WHERE id = ?
    ''', (labels.get("a", 0), article_ida))
    cursor.execute('''
        UPDATE titles
        SET label = ?
        WHERE id = ?
    ''', (labels.get("b", 0), article_idb))
    cursor.execute('''
        UPDATE titles
        SET label = ?
        WHERE id = ?
    ''', (labels.get("c", 0), article_idc))
    cursor.execute('''
        UPDATE titles
        SET label = ?
        WHERE id = ?
    ''', (labels.get("d", 0), article_idd))
    cursor.execute('''
        UPDATE titles
        SET label = ?
        WHERE id = ?
    ''', (labels.get("e", 0), article_ide))
    cursor.execute('''
        UPDATE titles
        SET label = ?
        WHERE id = ?
    ''', (labels.get("f", 0), article_idf))
    cursor.execute('''
        UPDATE titles
        SET label = ?
        WHERE id = ?
    ''', (labels.get("g", 0), article_idg))
    cursor.execute('''
        UPDATE titles
        SET label = ?
        WHERE id = ?
    ''', (labels.get("h", 0), article_idh))
    cursor.execute('''
        UPDATE titles
        SET label = ?
        WHERE id = ?
    ''', (labels.get("i", 0), article_idi))
    cursor.execute('''
        UPDATE titles
        SET label = ?
        WHERE id = ?
    ''', (labels.get("j", 0), article_idj))
    cursor.execute('''
        UPDATE titles
        SET label = ?
        WHERE id = ?
    ''', (labels.get("k", 0), article_idk))
    cursor.execute('''
        UPDATE titles
        SET label = ?
        WHERE id = ?
    ''', (labels.get("l", 0), article_idl))
    conn.commit()
    conn.close()

def labelArticles():
    rows = readDB()
    pbar = tqdm.tqdm(range(0, len(rows), 12), total=(len(rows)+11)//12, desc="Labeling Articles")
    for i in pbar:
        # 取出12篇，若不足則補最後一篇
        def get_row(idx):
            return rows[idx] if idx < len(rows) else rows[i]
        article_ida, _, texta = get_row(i)
        article_idb, _, textb = get_row(i+1)
        article_idc, _, textc = get_row(i+2)
        article_idd, _, textd = get_row(i+3)
        article_ide, _, texte = get_row(i+4)
        article_idf, _, textf = get_row(i+5)
        article_idg, _, textg = get_row(i+6)
        article_idh, _, texth = get_row(i+7)
        article_idi, _, texti = get_row(i+8)
        article_idj, _, textj = get_row(i+9)
        article_idk, _, textk = get_row(i+10)
        article_idl, _, textl = get_row(i+11)

        combined_text = (
            f"a. {texta}\nb. {textb}\nc. {textc}\nd. {textd}\n"
            f"e. {texte}\nf. {textf}\ng. {textg}\nh. {texth}\n"
            f"i. {texti}\nj. {textj}\nk. {textk}\nl. {textl}"
        )
        batch_num = (i // 12) + 1
        pbar.set_description(f"Labeling Batch {batch_num} ID: {article_ida}~{article_idl}")
        try:
            labels = generateStance(combined_text)
            if labels and isinstance(labels, dict):
                updateArticleLabels(
                    article_ida, article_idb, article_idc, article_idd,
                    article_ide, article_idf, article_idg, article_idh,
                    article_idi, article_idj, article_idk, article_idl, labels
                )
            else:
                tqdm.tqdm.write(f"No labels generated for article ID {article_ida}")
        except Exception as e:
            print(f"Error processing article: {e}")

if __name__ == "__main__":
    labelArticles()
    labelArticles()