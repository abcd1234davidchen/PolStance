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

    def connect(self):
        self.conn = sqlite3.connect(self.db_name, check_same_thread=False)
        self.cursor = self.conn.cursor()
        
        
    def readDB(self, label_name , batch_size: int = 12):
        """Read up to `batch_size` rows starting from offset `based`.

        Returns a tuple (rows, columns). Only returns rows where title and article
        are not NULL and at least one label is -1 (i.e. needs labeling).

        Parameters:
        - based (int): zero-based offset to start reading from (negative values are treated as 0)
        - batch_size (int): maximum number of rows to return (must be >=1)
        """
        # sanitize parameters
        

        try:
            limit = int(batch_size)
            if limit <= 0:
                raise ValueError("batch_size must be positive")
        except Exception as e:
            limit = 12
            print(f"Warning: Invalid 'batch_size' parameter: {e}")

        # Reads can be done without locking, but keep a short lock to be safe
        with self._lock:
            self.cursor.execute(
                f"""
                SELECT id, url, title, article, labelA, labelB, labelC, label FROM articleTable
                WHERE title IS NOT NULL
                    AND article IS NOT NULL
                    AND {label_name} = -1
                LIMIT ?
                """,
                (limit,)
            )
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
                #print(f"Updating id {data_id} with {label_name} = {value}")
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
