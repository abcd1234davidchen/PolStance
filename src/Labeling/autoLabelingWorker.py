
import traceback
import datetime
from pathlib import Path
from dotenv import load_dotenv
import tqdm
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from functools import partial
import signal
import sys
import os
import time 
from Labeling.geminiLabeling import GeminiLabeling
from Labeling.gptLabeling import GptLabeling
from Labeling.llamaLabeling import LlamaLabeling
from Labeling.utils.HFManager import HFManager
from Labeling.utils.DBManager import DBManager

class Color:
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    RESET = "\033[0m"

EXIT_KEY = False
TIME_OUT = int(os.getenv("TIMEOUT", "10"))
BATCH_SIZE = 12
def agentprocess(
    func,
    #bar: tqdm.tqdm,
    timeout,
) -> bool:
    """
    Execute a callable `func` in a thread and return its result dict.
    `func` should be a callable that returns dict[str,int].
    """
    result = True
    try:
        fut = _executor.submit(func)
        try:
            result = fut.result(timeout=timeout)
        except FuturesTimeoutError:
            result = False
            try:
                fut.cancel()
            except Exception:
                pass
            raise TimeoutError(f"Provider Timeout")
    except TimeoutError as te:
        raise TimeoutError from te
    except Exception as e:
        print(f"{type(func)} labeling error: {e}: {traceback.format_exc()}")
    return result


def labelArticles():
    geminiClient = GeminiLabeling()
    gptClient = GptLabeling()
    llamaClient = LlamaLabeling()
    # 共享 executor 避免每次建立 with-block 導致 shutdown(wait=True) 的阻塞
    global _executor
    if "_executor" not in globals():
        _executor = ThreadPoolExecutor(max_workers=3)
    # expose hf/db/executor for cleanup handlers
    global _hf, _db
    Path("tmp").mkdir(parents=True, exist_ok=True)
    hf = HFManager()
    db = hf.download_db()
    _hf = hf
    _db = db
    
    client_list = {
        "gemini": (geminiClient,"labelA"), 
        "gpt": (gptClient,"labelB"), 
        "llama": (llamaClient,"labelC")
    }
    
    for idx, (name, (client, label_col)) in enumerate(client_list.copy().items()):
        rows, columns = db.readDB(label_col, batch_size=1000000)
        length = len(rows)//12 + (len(rows)%12>0)
        bbar = tqdm.tqdm(
            range(length), desc=f"{label_col} progress"
        )
        for i in bbar:
            try:
                try:
                    func = partial(client.labeling_and_write_db, db, label_col, BATCH_SIZE)
                    agentprocess(func=func, timeout=TIME_OUT)
                except TimeoutError as te:
                    print(f"{Color.RED}{name} timeout error, skip client{Color.RESET}")
                    break
                    
                except Exception as e:
                    print(f"{name} setup error: {e}: {traceback.format_exc()}")
                
                time.sleep(2) 
            except Exception as e:
                print(f"Error processing article: {e}: {traceback.format_exc()}")

            if int(i)%100==0 and i>0:
                hf.upload_db("Update at " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                db.connect()
        hf.upload_db("Update at " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        db.connect()
    try:
        db.cursor.execute(f"""
            UPDATE 'articleTable'
            SET label = CASE
                WHEN labelA = labelB OR labelA = labelC THEN labelA
                WHEN labelB = labelC THEN labelB
                ELSE -2
            END
            WHERE labelA IS NOT NULL OR labelB IS NOT NULL OR labelC IS NOT NULL
        """)
        db.conn.commit()
        print("Vote aggregation completed successfully.")
    except Exception as e:
        print(f"Vote aggregation failed: {e}: {traceback.format_exc()}")
    db.close()
    hf.upload_db("Update at " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

if __name__ == "__main__":
    load_dotenv()

    def _cleanup(signum=None, frame=None):
        global _executor, _hf, _db
        _db.close()
        _executor.shutdown(wait=False)
        """Graceful shutdown for signals and atexit.

        If called from a signal handler, signum will be set and we exit.
        """
        print("Shutting down: cleaning up resources...")
        try:
            if "_executor" in globals() and _executor is not None:
                try:
                    _executor.shutdown(wait=False)
                except Exception as e:
                    print(f"Error shutting down executor: {e}")
        except Exception:
            pass

        try:
            if "_hf" in globals() and _hf is not None:
                try:
                    _hf.upload_db("Shutdown at " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                except Exception as e:
                    print(f"Error uploading DB on shutdown: {e}")
        except Exception as e :
            print(f"Error during HF cleanup: {e}")
        print("Cleanup complete.")
        sys.exit(0)

    signal.signal(signal.SIGINT, _cleanup)
    signal.signal(signal.SIGTERM, _cleanup)

    labelArticles()
