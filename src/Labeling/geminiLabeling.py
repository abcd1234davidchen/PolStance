from Labeling.mainClass import LabelingClass
import os
import json
import requests
import traceback
from pathlib import Path

class GeminiLabeling(LabelingClass):
    def __init__(self):
        super().__init__()
        self.model_id = "gemini-flash-latest"

    def _request_url(self):
        return f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_id}:generateContent?key={os.getenv('GEMINI_API_KEY')}"

    def _msg_template(self, prompt):
        return [{"role": "user", "parts": [{"text": prompt}]}]

    def _request_config(self, model, msg):
        return {
            "contents": msg,
            "system_instruction": {"parts": [{"text": self.instruction}]},
            "generation_config": {"temperature": 0.2},
        }

    def _requests_structure(self, config):
        return {
            "headers": {"Content-Type": "application/json; charset=utf-8"},
            "data": json.dumps(config),
        }

    def _send_request(self, prompt: list[dict[str, str]]):
        req_config = self._request_config(self.model_id, prompt)

        attempt = 0
        while attempt < self.MAX_ATTEMPTS:
            response = requests.post(
                url=self._request_url(), **(self._requests_structure(req_config))
            )
            if response.status_code != 200:
                attempt += 1
                continue
            else:
                response = response.text
            with open(
                f"tmp/{str(self.__class__.__name__)}_debug.json",
                "w",
            ) as f:
                f.write(str(response))
            res = self._get_response_text(json.loads(response))
            structured_response = self._parse_response(str(res))
            if structured_response is not None:
                return structured_response

            attempt += 1
        raise ValueError("Failed to get a valid response after maximum attempts")

    def _get_response_text(self, response: dict) -> str:
        try:
            res = response["candidates"][0]["content"]["parts"][0]["text"]
            res = res.replace("\'","\"")
        except (KeyError, IndexError):
            res = ""
            print(f"Warning: Unexpected response structure {traceback.format_exc()}")
            print(response)
        return res


if __name__ == "__main__":
    client = GeminiLabeling()
    client.labeling("台灣在哪裡？")
