
import traceback
import datetime
from pathlib import Path
from dotenv import load_dotenv
import tqdm
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from functools import partial
import signal
import sys
import time 
from Labeling.geminiLabeling import GeminiLabeling
from Labeling.gptLabeling import GptLabeling
from Labeling.llamaLabeling import LlamaLabeling
from Labeling.utils.HFManager import HFManager
from Labeling.utils.DBManager import DBManager


def agentprocess(
    func,
    #bar: tqdm.tqdm,
    timeout: int = 15,
) -> dict[str, int]:
    """
    Execute a callable `func` in a thread and return its result dict.
    `func` should be a callable that returns dict[str,int].
    """
    
    try:
        fut = _executor.submit(func)
        try:
            labels = fut.result(timeout=timeout)
        except FuturesTimeoutError:
            print(f"{getattr(func, '__name__', str(func))} labeling timeout after {timeout}s")
            labels = {}
            try:
                fut.cancel()
            except Exception:
                pass
            raise TimeoutError(f"Provider Timeout, consider disabling {getattr(func, '__name__', str(func))}")
    except TimeoutError as te:
        print(f"{type(func)} labeling timeout error: {te}: {traceback.format_exc()}")
        raise TimeoutError from te
    except Exception as e:
        print(f"{type(func)} labeling error: {e}: {traceback.format_exc()}")
        labels = {}
    #bar.update(1)
    return labels


def compare_labels(result_dict: dict[str, dict[str, int]]) -> dict[str, int]:
    final_labels = {}
    for key in result_dict["gemini"].keys():
        votes = {}
        for model in result_dict.keys():
            label = result_dict[model].get(key, 0)
            votes[label] = votes.get(label, 0) + 1
        final_label = max(votes.items(), key=lambda x: x[1])[0]
        final_labels[key] = final_label
    return final_labels


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

    rows, columns = db.readDB()
    
    bbar = tqdm.tqdm(
        range(0, len(rows), 12), total=(len(rows) + 11) // 12, desc="Batch progress"
    )
    client_list = {"gemini": geminiClient, "gpt": gptClient, "llama": llamaClient}
    for i in bbar:
        #pbar = tqdm.tqdm(total=len(client_list.copy()), leave=False, desc="Batch Labeling")
        
        try:
            tasks = {}
            for idx, (name, client) in enumerate(client_list.copy().items()):
                try:
                    label_col = f"label{chr(ord('A')+idx)}"
                    func = partial(client.labeling_and_write_db, db, label_col, 12)
                    tasks[name] = agentprocess(func, timeout=10)

                except TimeoutError as te:
                    print(f"{name} timeout error")
                    print("disabling this client for future tasks.")
                    client_list.pop(name)
                    
                except Exception as e:
                    print(f"{name} setup error: {e}: {traceback.format_exc()}")
                time.sleep(2) 
        except Exception as e:
            print(f"Error processing article: {e}: {traceback.format_exc()}")

    #hf.upload_db("Update at " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    load_dotenv()

    def _cleanup(signum=None, frame=None):
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

        # try:
        #     if "_hf" in globals() and _hf is not None:
        #         try:
        #             _hf.upload_db("Shutdown at " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        #         except Exception as e:
        #             print(f"Error uploading DB on shutdown: {e}")
        # except Exception:
        #     pass

        try:
            if "_db" in globals() and _db is not None:
                try:
                    _db.close()
                except Exception as e:
                    print(f"Error closing DB: {e}")
        except Exception:
            pass
        print("Cleanup complete.")
        sys.exit(0)

    signal.signal(signal.SIGINT, _cleanup)
    signal.signal(signal.SIGTERM, _cleanup)

    labelArticles()
