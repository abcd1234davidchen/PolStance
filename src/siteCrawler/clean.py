# ...existing code...
#!/usr/bin/env python3
import sqlite3
import shutil
import sys
import os

PHRASES = [
    "【更多內容 請見影片】訂閱【自由追新聞】全新的視界！新聞話題不漏接，快訂閱YouTube 【自由追新聞】，記得開啟小鈴鐺哦！",
    "訂閱【自由追新聞】全新的視界！新聞 話題不漏接，快訂閱YouTube 【自由追新聞】，記得開啟小鈴鐺哦！",
    "// 創物件 var tvPlayer = new VideoAPI_LiTV(); // 設定自動播放 tvPlayer.setAutoplay(true); //不自動播放 tvPlayer.setDelay(0); // 設定延遲 tvPlayer.setAllowFullscreen(true); tvPlayer.setType('web'); // tvPlayer.setControls(1); litv 無法操作顯示控制項 tvPlayer.pushVideoIdByClassName('TVPlayer', tvPlayer); setTimeout(function (){ tvPlayer.loadAPIScript('cache_video_js_LiTV'); },3000)",
    "©",
    "This is a modal window",
    "點擊免費加入會員！",
    "【加入關鍵評論網會員】",
    "每天精彩好文直送你的信箱，每週獨享編輯精選、時事精選、藝文週報等特製電子報。還可留言與作者、記者、編輯討論文章內容。",
    "立刻點擊免費加入會員。",
]

def text_columns(conn, table):
    cur = conn.execute(f"PRAGMA table_info({table})")
    cols = []
    for cid, name, ctype, notnull, dflt, pk in cur.fetchall():
        if ctype is None:
            continue
        t = ctype.upper()
        if any(k in t for k in ("CHAR", "CLOB", "TEXT")):
            cols.append(name)
    return cols

def tables(conn):
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
    return [r[0] for r in cur.fetchall()]

def clean_db(db_path):
    conn = sqlite3.connect(db_path)
    conn.isolation_level = None
    try:
        conn.execute("BEGIN")
        total_updates = 0
        for tbl in tables(conn):
            cols = text_columns(conn, tbl)
            if not cols:
                continue
            for col in cols:
                for phrase in PHRASES:
                    sql = f"UPDATE {tbl} SET {col} = REPLACE({col}, ?, '') WHERE {col} LIKE '%' || ? || '%'"
                    cur = conn.execute(sql, (phrase, phrase))
                    total_updates += cur.rowcount
        conn.execute("COMMIT")
        print("完成。總更新筆數 (受影響的儲存格數):", total_updates)
    except Exception as e:
        conn.execute("ROLLBACK")
        print("錯誤，已回滾：", e)
    finally:
        conn.close()

def remove_row(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('DELETE FROM articleTable WHERE title IS NULL OR title = ""')
    cursor.execute('SELECT id,title FROM articleTable WHERE title IS NOT NULL AND title != ""')
    titles = cursor.fetchall()
    for title_id, title in titles:
        if title is None or title=="":
            continue
        if len(title.strip()) < 18:
            cursor.execute('DELETE FROM articleTable WHERE id = ?', (title_id,))
    conn.commit()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM articleTable WHERE article = "?"')
    cursor.execute('SELECT id,article FROM articleTable WHERE article IS NOT NULL AND article != "?"')
    articles = cursor.fetchall()
    for article_id, article in articles:
        if article is None or article=="":
            continue
        if len(article.strip()) < 50:
            cursor.execute('DELETE FROM articleTable WHERE id = ?', (article_id,))
    conn.commit()
    conn.close()

def main():
    clean_db("article.db")
    # remove_row("article.db")

if __name__ == "__main__":
    main()