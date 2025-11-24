from dotenv import load_dotenv
import json
import random
import requests
import os
from typing import Any
import traceback

class LabelingClass:
    model_id: str | None = None
    ENDPOINT: str | None = None
    REGION: str | None = None

    MAX_ATTEMPTS = 1
    return_structure = {chr(ord("a") + i): random.randint(1, 3) for i in range(12)}
    return_structure_type = {chr(ord("a") + i): "int" for i in range(12)}

    instruction = f"""
你是一個政治立場分析專家，並會強烈依照使用者的指示回答問題。現在，請根據使用者提供的多個文章，分析每個文章的政治立場傾向，並以JSON格式回覆。
若傾向國民黨則標示為\"1\"，若傾向民進黨則標示為\"2\"，若為中立則標示為\"3\"
每篇文章只能有一個傾向，並且僅能夠從上述的三種種類中挑選。
並嚴格遵照下列的結構輸出結果：
{str(return_structure_type)}

舉例:
{str(return_structure)}

你僅需要輸出這個JSON結構，請勿添加任何額外的文字或說明。
"""

    def __init__(self) -> None:
        load_dotenv()

    def _msg_template(self, prompt) -> list[Any]:
        return [
            {"role": "system", "content": self.instruction},
            {"role": "user", "content": prompt},
        ]

    def _request_config(self, model, msg) -> dict[str, Any]:
        return {
            "model": model,
            "messages": msg,
            "temperature": 0,
            "provider": {
                "allow_fallbacks": True,
            }
        }

    def _requests_structure(self, config) -> dict[str, Any]:
        token = os.getenv("OPR_KEY")
        return {
            "headers": {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            },
            "data": json.dumps(config),
        }

    def _request_url(self):
        return f"https://openrouter.ai/api/v1/chat/completions"
    
    def _get_response_text(self, response: dict) -> str:
        try:
            res = response["choices"][0]["message"]["content"]
            res = res.replace("```json", "").replace("```", "").replace('\'', '"')
            
        except (KeyError, IndexError):
            res = ""
            print(response)
            print(f"Warning: Unexpected response structure. {traceback.format_exc()}")
        return res

    def _parse_response(self, response: str) -> dict[str, int] | None:
        for replacement in ["`", " ", "\n", "json"]:
            response = response.replace(replacement, "")
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            print(f"JSON decode error at {self.__class__.__name__}: {e}")
            return None

    def labeling(self, prompt: str) -> dict[str, int]:
        if hasattr(self, "_send_request"):
            structured_response = self._send_request(self._msg_template(prompt=prompt))
        else:
            raise NotImplementedError("_send_request method not implemented.")
        return structured_response

    def labeling_and_write_db(self, db_manager, label_name: str, batch_size: int = 12) -> bool:
        """
        Convenience method: read up to `batch_size` rows from `db_manager`, build the
        combined prompt, call `labeling`, then write results back to DB using
        `db_manager.updateArticleLabels`.

        Returns the labeling result (dict) or empty dict on failure.
        """
        rows, columns = db_manager.readDB(label_name, batch_size)
        if not rows:
            return False

        # Prepare row_data similar to how autoLabelingWorker used it: mapping offset->row
        def get_row(idx):
            return rows[idx] if idx < len(rows) else rows[0]

        row_data = {}
        for offset in range(0, batch_size):
            row_data[offset] = get_row(offset)

        # Build combined prompt using the 'title' column
        try:
            title_index = columns.index("title")
        except ValueError:
            title_index = 2

        combined_text = "\n".join(
            [f"{(chr(ord('a') + idx))}. {row[title_index]}" for idx, row in enumerate(row_data.values())]
        )

        try:
            result = self.labeling(combined_text)
            if result and isinstance(result, dict):
                # Write labels back to DB
                db_manager.updateArticleLabels(label_name, row_data, result)
                return True
            else:
                print(f"{self.__class__.__name__} received invalid labeling result.")
        except Exception as e:
            print(f"{self.__class__.__name__} labeling error: {e}: {traceback.format_exc()}")

        return True

    def _send_request(self, prompt: list[dict[str, str]]):
        req_config = self._request_config(self.model_id, prompt)

        attempt = 0
        while attempt < self.MAX_ATTEMPTS:
            response = requests.post(
                url=self._request_url(), **(self._requests_structure(req_config))
            )
            if response.status_code != 200:
                attempt += 1
                print(f"request failed, retrying.... Error:{response}")
                print(response.text)
                continue
            else:
                response_data = response.json()
            
            with open(
                f"tmp/{str(self.__class__.__name__)}_debug.json",
                "w+",
                encoding="utf-8",
            ) as f:
                json.dump(response_data, f, ensure_ascii=False, indent=2)
            try:
                model_content = self._get_response_text(response_data)
            except Exception as e:
                print(response_data)
                print(f"Error extracting response text: {e}")
                attempt += 1
                continue
            structured_response = self._parse_response(model_content)
            if structured_response is not None:
                return structured_response
            else:
                print(f"Parsing failed, retrying... Response was: {model_content}")
                attempt += 1
        raise ValueError(f"Failed to get a valid response after maximum attempts")
