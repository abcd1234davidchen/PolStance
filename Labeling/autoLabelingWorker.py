from dotenv import load_dotenv
import tqdm

from geminiLabeling import GeminiLabeling
from gptLabeling import GptLabeling
from llamaLabeling import LlamaLabeling
from utils.DBManager import DbManager
from utils.HFManager import HFManager
import traceback

geminiClient = GeminiLabeling()
gptClient = GptLabeling()
llamaClient = LlamaLabeling()

hf = HFManager()
db = hf.download_db()

def gpt_process(combined_text: str, bar:tqdm.tqdm) -> dict[str, int]:
    try:
        labels_gpt = gptClient.labeling(combined_text)
        #print(f"GPT Labels: {labels_gpt}")
        
    except Exception as e:
        print(f"GPT labeling error: {e}: {traceback.format_exc()}")
        labels_gpt = {}
    bar.update(1)
    return labels_gpt

def gemini_process(combined_text: str, bar:tqdm.tqdm) -> dict[str, int]:
    try:
        labels_gemini = geminiClient.labeling(combined_text)
        #print(f"Gemini Labels: {labels_gemini}")
    except Exception as e:
        print(f"Gemini labeling error: {e}: {traceback.format_exc()}")
        labels_gemini = {}
    bar.update(1)
    return labels_gemini

def llama_process(combined_text: str, bar:tqdm.tqdm) -> dict[str, int]:
    try:
        labels_ds = llamaClient.labeling(combined_text)
        #print(f"llama Labels: {labels_ds}")
    except Exception as e:
        print(f"llama labeling error: {e}: {traceback.format_exc()}")
        labels_ds = {}
    bar.update(1)
    return labels_ds

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
    rows = db.readDB()
    bbar = tqdm.tqdm(
        range(0, len(rows), 12), total=(len(rows) + 11) // 12, desc="Batch progress"
    )
    for i in bbar:
        # 取出12篇，若不足則補最後一篇
        pbar = tqdm.tqdm(range(0, 3), leave=False, total=3, desc="Batch Labeling")
        def get_row(idx):
            return rows[idx] if idx < len(rows) else rows[i]

        row_data = {}
        for offset in range(0, 12):
            data_id, url, title, article = get_row(i + offset)
            row_data[offset] = (data_id, url, title, article)

        combined_text = "\n".join(
            [
                f"{(chr(ord('a') + idx))}. {title}"
                for idx, (data_id, _, title, article) in enumerate(row_data.values())
            ]
        )
        # "a. title1\n b. title2\n c. title3\n ..."
        print("Combined Text for labeling:\n" + combined_text)
        batch_num = (i // 12) + 1
        bbar.set_description(
            f"Labeling Batch {batch_num} ID: {row_data[0][0]}~{row_data[11][0]}"
        )
        try:
            labeling_result = {
                "gemini":gemini_process(combined_text, pbar),
                "gpt" : gpt_process(combined_text, pbar),
                "llama":llama_process(combined_text, pbar)
            }
            input()
            print(f"Labeling Result: {labeling_result}")
            voted_labels = compare_labels(labeling_result)
            db.updateArticleLabels(row_data, labeling_result, voted_labels)
            pbar.close() 
        except Exception as e:
            print(f"Error processing article: {e}: {traceback.format_exc()}")

    

if __name__ == "__main__":
    load_dotenv()
    labelArticles()


