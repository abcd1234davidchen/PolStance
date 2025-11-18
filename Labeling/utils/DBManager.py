import sqlite3


class DbManager:
    def __init__(self, db_name='article.db'): 
        self.db_name = db_name
        self.conn = sqlite3.connect(self.db_name)
        self.cursor = self.conn.cursor()

    def readDB(self,):
        self.cursor.execute("""
            SELECT id, url, title, article FROM articleTable 
            WHERE title IS NOT NULL AND title != '' 
        """)
        rows = self.cursor.fetchall()
        return rows

    def updateArticleLabels(self, row_datas, model_labels, voted_labels):
        for idx, (data_id, url, title, article) in enumerate(row_datas.values()):
            self.cursor.execute('''
                UPDATE articleTable
                SET labelA = ?, labelB = ?, labelC = ?
                WHERE id = ?
            ''', (
                model_labels.get("gemini", {}).get(f"{chr(ord('a')+idx)}", -1),
                model_labels.get("gpt",{}).get(f"{chr(ord('a')+idx)}", -1),
                model_labels.get("llama", {}).get(f"{chr(ord('a')+idx)}", -1),
                data_id
            ))
            self.cursor.execute('''
                UPDATE articleTable
                SET label = ?
                WHERE id = ? 
                  AND labelA NOT IN (-1, -2) 
                  AND labelB NOT IN (-1, -2) 
                  AND labelC NOT IN (-1, -2)
            ''', (
                voted_labels.get(f"{chr(ord('a')+idx)}", -1),
                data_id
            ))
        self.conn.commit()
    
    def close(self):
        self.conn.close()
        
    def __del__(self):
        self.close()
    
    
    