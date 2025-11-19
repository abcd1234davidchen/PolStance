import sqlite3
import threading
import re


class DBManager:
    def __init__(self, db_name="article.db"):
        self.db_name = db_name
        # allow usage from multiple threads; we'll guard writes with a lock
        self.conn = sqlite3.connect(self.db_name, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self._lock = threading.Lock()

    def readDB(self):
        # Reads can be done without locking, but keep a short lock to be safe
        with self._lock:
            self.cursor.execute("""
                SELECT id, url, title, article, labelA, labelB, labelC, label FROM articleTable 
                WHERE title IS NOT NULL 
                    AND article IS NOT NULL 
                    AND (labelA = -1 OR labelB = -1 OR labelC = -1 OR label = -1)
            """)
            rows = self.cursor.fetchall()
        return rows, ("id", "url", "title", "article", "labelA", "labelB", "labelC", "label")

    def updateArticleLabels(self, label_name, row_datas, model_labels):
        # Validate label_name to avoid SQL injection (must be labelA/labelB/labelC or label)
        if not re.match(r"^label[A-C]$|^label$", label_name):
            raise ValueError(f"Invalid label column: {label_name}")

        with self._lock:
            for idx, row in enumerate(row_datas.values()):
                # each row is expected to be a tuple where first element is id
                data_id = row[0]
                value = model_labels.get(f"{chr(ord('a') + idx)}", -1)
                # column name can't be parameterized, use validated label_name
                sql = f"UPDATE articleTable SET {label_name} = ? WHERE id = ?"
                self.cursor.execute(sql, (value, data_id))
            self.conn.commit()

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass

    def __del__(self):
        self.close()
